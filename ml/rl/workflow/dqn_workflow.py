#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import sys
from typing import Dict

import numpy as np
import torch
from ml.rl.evaluation.evaluator import Evaluator
from ml.rl.json_serialize import from_json
from ml.rl.models.output_transformer import DiscreteActionOutputTransformer
from ml.rl.parameters import (
    DiscreteActionModelParameters,
    NormalizationParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.preprocessing.batch_preprocessor import DiscreteDqnBatchPreprocessor
from ml.rl.preprocessing.feature_extractor import PredictorFeatureExtractor
from ml.rl.preprocessing.normalization import sort_features_by_normalization
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.preprocessing.sparse_to_dense import PandasSparseToDenseProcessor
from ml.rl.readers.json_dataset_reader import JSONDatasetReader
from ml.rl.tensorboardX import summary_writer_context
from ml.rl.training.dqn_trainer import DQNTrainer
from ml.rl.training.rl_exporter import DQNExporter
from ml.rl.workflow.base_workflow import BaseWorkflow
from ml.rl.workflow.helpers import (
    export_trainer_and_predictor,
    minibatch_size_multiplier,
    parse_args,
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
        model_params = model_params

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

    model_params = DiscreteActionModelParameters(
        actions=action_names,
        rl=rl_parameters,
        training=training_parameters,
        rainbow=rainbow_parameters,
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

    sorted_features, _ = sort_features_by_normalization(state_normalization)
    preprocess_handler = DiscreteDqnPreprocessHandler(
        action_names, PandasSparseToDenseProcessor(sorted_features)
    )

    train_dataset = JSONDatasetReader(
        params["training_data_path"],
        batch_size=training_parameters.minibatch_size,
        preprocess_handler=preprocess_handler,
    )
    eval_dataset = JSONDatasetReader(
        params["eval_data_path"],
        batch_size=training_parameters.minibatch_size,
        preprocess_handler=preprocess_handler,
    )

    with summary_writer_context(writer):
        workflow.train_network(train_dataset, eval_dataset, int(params["epochs"]))

    exporter = DQNExporter(
        workflow.trainer.q_network,
        PredictorFeatureExtractor(state_normalization_parameters=state_normalization),
        DiscreteActionOutputTransformer(model_params.actions),
    )

    if int(params["node_index"]) == 0 and gpu_index == 0:
        export_trainer_and_predictor(
            workflow.trainer, params["model_output_path"], exporter=exporter
        )  # noqa


def main(params):
    if "node_index" not in params or params["node_index"] is None:
        params["node_index"] = 0
    can_do_distributed = torch.distributed.is_available() and torch.cuda.is_available()
    if params["use_all_avail_gpus"] and not can_do_distributed:
        logger.info(
            "Horizon is configured to use all GPUs but your platform doesn't support torch.distributed & torch.cuda!"
        )
        params["use_all_avail_gpus"] = False
    if params["use_all_avail_gpus"]:
        params["num_processes_per_node"] = max(1, torch.cuda.device_count())
        multiprocessing.spawn(
            single_process_main, nprocs=params["num_processes_per_node"], args=[params]
        )
    else:
        single_process_main(0, params)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    params = parse_args(sys.argv)
    main(params)
