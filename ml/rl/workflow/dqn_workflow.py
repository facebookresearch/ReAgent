#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import os
import sys
import time
from typing import Dict

import numpy as np
import torch
from ml.rl.evaluation.evaluator import Evaluator
from ml.rl.json_serialize import from_json
from ml.rl.parameters import (
    DiscreteActionModelParameters,
    EvaluationParameters,
    NormalizationParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.prediction.predictor_wrapper import (
    DiscreteDqnPredictorWrapper,
    DiscreteDqnWithPreprocessor,
)
from ml.rl.preprocessing.batch_preprocessor import DiscreteDqnBatchPreprocessor
from ml.rl.preprocessing.normalization import sort_features_by_normalization
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.preprocessing.sparse_to_dense import StringKeySparseToDenseProcessor
from ml.rl.readers.json_dataset_reader import JSONDatasetReader
from ml.rl.tensorboardX import summary_writer_context
from ml.rl.training.dqn_trainer import DQNTrainer
from ml.rl.workflow.base_workflow import BaseWorkflow
from ml.rl.workflow.helpers import (
    minibatch_size_multiplier,
    parse_args,
    save_model_to_file,
    update_model_for_warm_start,
)
from ml.rl.workflow.preprocess_handler import DiscreteDqnPreprocessHandler
from ml.rl.workflow.transitional import create_dqn_trainer_from_params
from torch import multiprocessing
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


class DqnWorkflow(BaseWorkflow):
    def __init__(
        self,
        model_params: DiscreteActionModelParameters,
        state_normalization: Dict[int, NormalizationParameters],
        use_gpu: bool,
        use_all_avail_gpus: bool,
    ):
        logger.info("Running DQN workflow with params:")
        logger.info(model_params)
        self.model_params = model_params
        self.state_normalization = state_normalization

        trainer = create_dqn_trainer_from_params(
            model_params,
            state_normalization,
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
        )
        trainer = update_model_for_warm_start(trainer)
        assert type(trainer) == DQNTrainer, "Warm started wrong model type: " + str(
            type(trainer)
        )

        evaluator = Evaluator(
            model_params.actions,
            model_params.rl.gamma,
            trainer,
            metrics_to_score=trainer.metrics_to_score,
        )

        super().__init__(
            DiscreteDqnBatchPreprocessor(Preprocessor(state_normalization, use_gpu)),
            trainer,
            evaluator,
            model_params.training.minibatch_size,
        )

    def save_models(self, path: str):
        dqn_with_preprocessor = DiscreteDqnWithPreprocessor(
            self.trainer.q_network.cpu_model().eval(),
            Preprocessor(self.state_normalization, False),
        )
        serving_module = DiscreteDqnPredictorWrapper(
            dqn_with_preprocessor=dqn_with_preprocessor,
            action_names=self.model_params.actions,
        )

        export_time = round(time.time())
        output_path = os.path.expanduser(path)
        pytorch_output_path = os.path.join(output_path, f"trainer_{export_time}.pt")
        torchscript_output_path = os.path.join(
            path, "model_{}.torchscript".format(export_time)
        )
        logger.info("Saving PyTorch trainer to {}".format(pytorch_output_path))
        save_model_to_file(self.trainer, pytorch_output_path)
        self.save_torchscript_model(serving_module, torchscript_output_path)


def single_process_main(gpu_index, *args):
    params = args[0]
    # Set minibatch size based on # of devices being used to train
    params["training"]["minibatch_size"] *= minibatch_size_multiplier(
        params["use_gpu"], params["use_all_avail_gpus"]
    )

    action_names = params["actions"]

    rl_parameters = from_json(params["rl"], RLParameters)
    training_parameters = from_json(params["training"], TrainingParameters)
    rainbow_parameters = from_json(params["rainbow"], RainbowDQNParameters)
    if "evaluation" in params:
        evaluation_parameters = from_json(params["evaluation"], EvaluationParameters)
    else:
        evaluation_parameters = EvaluationParameters()

    model_params = DiscreteActionModelParameters(
        actions=action_names,
        rl=rl_parameters,
        training=training_parameters,
        rainbow=rainbow_parameters,
        evaluation=evaluation_parameters,
    )
    state_normalization = BaseWorkflow.read_norm_file(params["state_norm_data_path"])

    writer = SummaryWriter(log_dir=params["model_output_path"])
    logger.info("TensorBoard logging location is: {}".format(writer.log_dir))

    if params["use_all_avail_gpus"]:
        BaseWorkflow.init_multiprocessing(
            int(params["num_processes_per_node"]),
            int(params["num_nodes"]),
            int(params["node_index"]),
            gpu_index,
            params["init_method"],
        )

    workflow = DqnWorkflow(
        model_params,
        state_normalization,
        params["use_gpu"],
        params["use_all_avail_gpus"],
    )

    from os.path import abspath
    from petastorm import make_reader, TransformSpec
    from petastorm.pytorch import DataLoader
    from petastorm.spark_utils import dataset_as_rdd
    from convert_timeline_to_petastorm import get_spark_session

    train_url = f"file://{abspath(params['training_data_petastorm_path'])}"
    eval_url = f"file://{abspath(params['eval_data_petastorm_path'])}"

    sorted_features, _ = sort_features_by_normalization(state_normalization)
    preprocess_handler = DiscreteDqnPreprocessHandler(
        len(action_names), StringKeySparseToDenseProcessor(sorted_features)
    )

    epochs = int(params["epochs"])
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch} of {epochs}")

        # make readers one epoch and iterate through them n times
        train_reader = make_reader(train_url, num_epochs=1)
        eval_reader = make_reader(eval_url, num_epochs=1)

        with DataLoader(
            train_reader, batch_size=training_parameters.minibatch_size
        ) as train_dataloader, summary_writer_context(writer):
            workflow.dataloader_train_network(train_dataloader, epoch)

    if int(params["node_index"]) == 0 and gpu_index == 0:
        workflow.save_models(params["model_output_path"])


def main(params):
    if "node_index" not in params or params["node_index"] is None:
        params["node_index"] = 0
    can_do_distributed = torch.distributed.is_available() and torch.cuda.is_available()
    if params["use_all_avail_gpus"] and not can_do_distributed:
        logger.info(
            "Horizon is configured to use all GPUs but your platform doesn't support torch.distributed & torch.cuda!"
        )
        params["use_all_avail_gpus"] = False
    if params["use_gpu"] and not torch.cuda.is_available():
        logger.info("GPU requested but not available")
        params["use_gpu"] = False

    if params["use_all_avail_gpus"]:
        params["num_processes_per_node"] = max(1, torch.cuda.device_count())
        multiprocessing.spawn(
            single_process_main, nprocs=params["num_processes_per_node"], args=[params]
        )
    else:
        single_process_main(0, params)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)
    params = parse_args(sys.argv)
    main(params)
