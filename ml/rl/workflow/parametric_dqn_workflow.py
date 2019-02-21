#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import sys
from typing import Dict

from ml.rl.evaluation.evaluator import Evaluator
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.preprocessing.sparse_to_dense import PandasSparseToDenseProcessor
from ml.rl.readers.json_dataset_reader import JSONDatasetReader
from ml.rl.tensorboardX import summary_writer_context
from ml.rl.thrift.core.ttypes import (
    ContinuousActionModelParameters,
    NormalizationParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.training.parametric_dqn_trainer import ParametricDQNTrainer
from ml.rl.workflow.base_workflow import BaseWorkflow
from ml.rl.workflow.helpers import (
    export_trainer_and_predictor,
    minibatch_size_multiplier,
    parse_args,
    update_model_for_warm_start,
)
from ml.rl.workflow.preprocess_handler import (
    ParametricDqnPreprocessHandler,
    PreprocessHandler,
)
from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class ParametricDqnWorkflow(BaseWorkflow):
    def __init__(
        self,
        model_params: ContinuousActionModelParameters,
        preprocess_handler: PreprocessHandler,
        state_normalization: Dict[int, NormalizationParameters],
        action_normalization: Dict[int, NormalizationParameters],
        use_gpu: bool,
        use_all_avail_gpus: bool,
    ):
        logger.info("Running Parametric DQN workflow with params:")
        logger.info(model_params)
        model_params = model_params

        trainer = ParametricDQNTrainer(
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

        super(ParametricDqnWorkflow, self).__init__(
            preprocess_handler, trainer, evaluator, model_params.training.minibatch_size
        )


def main(params):
    # Set minibatch size based on # of devices being used to train
    params["training"]["minibatch_size"] *= minibatch_size_multiplier(
        params["use_gpu"], params["use_all_avail_gpus"]
    )

    rl_parameters = RLParameters(**params["rl"])
    training_parameters = TrainingParameters(**params["training"])
    rainbow_parameters = RainbowDQNParameters(**params["rainbow"])

    model_params = ContinuousActionModelParameters(
        rl=rl_parameters, training=training_parameters, rainbow=rainbow_parameters
    )
    state_normalization = BaseWorkflow.read_norm_file(params["state_norm_data_path"])
    action_normalization = BaseWorkflow.read_norm_file(params["action_norm_data_path"])

    writer = SummaryWriter(log_dir=params["model_output_path"])
    logger.info("TensorBoard logging location is: {}".format(writer.log_dir))

    preprocess_handler = ParametricDqnPreprocessHandler(
        Preprocessor(state_normalization, False),
        Preprocessor(action_normalization, False),
        PandasSparseToDenseProcessor(),
    )

    workflow = ParametricDqnWorkflow(
        model_params,
        preprocess_handler,
        state_normalization,
        action_normalization,
        params["use_gpu"],
        params["use_all_avail_gpus"],
    )

    train_dataset = JSONDatasetReader(
        params["training_data_path"], batch_size=training_parameters.minibatch_size
    )
    eval_dataset = JSONDatasetReader(params["eval_data_path"], batch_size=16)

    with summary_writer_context(writer):
        workflow.train_network(train_dataset, eval_dataset, int(params["epochs"]))
    return export_trainer_and_predictor(
        workflow.trainer, params["model_output_path"]
    )  # noqa


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    params = parse_args(sys.argv)
    main(params)
