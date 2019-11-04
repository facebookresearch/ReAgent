#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import os
import sys
import time
from typing import Dict

import torch
from ml.rl.evaluation.evaluator import Evaluator
from ml.rl.json_serialize import from_json
from ml.rl.parameters import (
    ContinuousActionModelParameters,
    NormalizationParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.prediction.predictor_wrapper import (
    ParametricDqnPredictorWrapper,
    ParametricDqnWithPreprocessor,
)
from ml.rl.preprocessing.batch_preprocessor import ParametricDqnBatchPreprocessor
from ml.rl.preprocessing.normalization import sort_features_by_normalization
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.preprocessing.sparse_to_dense import PandasSparseToDenseProcessor
from ml.rl.readers.json_dataset_reader import JSONDatasetReader
from ml.rl.tensorboardX import summary_writer_context
from ml.rl.torch_utils import export_module_to_buffer
from ml.rl.training.parametric_dqn_trainer import ParametricDQNTrainer
from ml.rl.workflow.base_workflow import BaseWorkflow
from ml.rl.workflow.helpers import (
    minibatch_size_multiplier,
    parse_args,
    save_model_to_file,
    update_model_for_warm_start,
)
from ml.rl.workflow.preprocess_handler import (
    ParametricDqnPreprocessHandler,
    PreprocessHandler,
)
from ml.rl.workflow.transitional import create_parametric_dqn_trainer_from_params
from torch import multiprocessing
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class ParametricDqnWorkflow(BaseWorkflow):
    def __init__(
        self,
        model_params: ContinuousActionModelParameters,
        state_normalization: Dict[int, NormalizationParameters],
        action_normalization: Dict[int, NormalizationParameters],
        use_gpu: bool,
        use_all_avail_gpus: bool,
    ):
        logger.info("Running Parametric DQN workflow with params:")
        logger.info(model_params)
        self.model_params = model_params
        self.state_normalization = state_normalization
        self.action_normalization = action_normalization

        trainer = create_parametric_dqn_trainer_from_params(
            model_params,
            state_normalization,
            action_normalization,
            use_gpu=use_gpu,
            use_all_avail_gpus=use_all_avail_gpus,
        )
        trainer = update_model_for_warm_start(trainer)
        assert (
            type(trainer) == ParametricDQNTrainer
        ), "Warm started wrong model type: " + str(type(trainer))

        evaluator = Evaluator(
            None,
            model_params.rl.gamma,
            trainer,
            metrics_to_score=trainer.metrics_to_score,
        )

        super().__init__(
            ParametricDqnBatchPreprocessor(
                Preprocessor(state_normalization, use_gpu),
                Preprocessor(action_normalization, use_gpu),
            ),
            trainer,
            evaluator,
            model_params.training.minibatch_size,
        )

    def save_models(self, path: str):
        export_time = round(time.time())
        output_path = os.path.expanduser(path)
        pytorch_output_path = os.path.join(
            output_path, "trainer_{}.pt".format(export_time)
        )
        torchscript_output_path = os.path.join(
            path, "model_{}.torchscript".format(export_time)
        )

        state_preprocessor = Preprocessor(self.state_normalization, False)
        action_preprocessor = Preprocessor(self.action_normalization, False)
        q_network = self.trainer.q_network
        dqn_with_preprocessor = ParametricDqnWithPreprocessor(
            q_network.cpu_model().eval(), state_preprocessor, action_preprocessor
        )
        serving_module = ParametricDqnPredictorWrapper(
            dqn_with_preprocessor=dqn_with_preprocessor
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

    rl_parameters = from_json(params["rl"], RLParameters)
    training_parameters = from_json(params["training"], TrainingParameters)
    rainbow_parameters = from_json(params["rainbow"], RainbowDQNParameters)

    model_params = ContinuousActionModelParameters(
        rl=rl_parameters, training=training_parameters, rainbow=rainbow_parameters
    )
    state_normalization = BaseWorkflow.read_norm_file(params["state_norm_data_path"])
    action_normalization = BaseWorkflow.read_norm_file(params["action_norm_data_path"])

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

    workflow = ParametricDqnWorkflow(
        model_params,
        state_normalization,
        action_normalization,
        params["use_gpu"],
        params["use_all_avail_gpus"],
    )

    state_sorted_features, _ = sort_features_by_normalization(state_normalization)
    action_sorted_features, _ = sort_features_by_normalization(action_normalization)
    preprocess_handler = ParametricDqnPreprocessHandler(
        PandasSparseToDenseProcessor(state_sorted_features),
        PandasSparseToDenseProcessor(action_sorted_features),
    )

    train_dataset = JSONDatasetReader(
        params["training_data_path"],
        batch_size=training_parameters.minibatch_size,
        preprocess_handler=preprocess_handler,
    )
    eval_dataset = JSONDatasetReader(
        params["eval_data_path"], batch_size=16, preprocess_handler=preprocess_handler
    )

    with summary_writer_context(writer):
        workflow.train_network(train_dataset, eval_dataset, int(params["epochs"]))

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
