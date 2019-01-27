#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import sys
import time

import numpy as np
from ml.rl.evaluation.evaluation_data_page import EvaluationDataPage
from ml.rl.evaluation.evaluator import Evaluator
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.thrift.core.ttypes import (
    ContinuousActionModelParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.training.parametric_dqn_trainer import ParametricDQNTrainer
from ml.rl.workflow.helpers import (
    export_trainer_and_predictor,
    minibatch_size_multiplier,
    parse_args,
    report_training_status,
    update_model_for_warm_start,
)
from ml.rl.workflow.training_data_reader import (
    JSONDataset,
    preprocess_batch_for_training,
    read_norm_file,
)


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

DEFAULT_NUM_SAMPLES_FOR_CPE = 500


def train_network(params):
    logger.info("Running Parametric DQN workflow with params:")
    logger.info(params)

    # Set minibatch size based on # of devices being used to train
    params["training"]["minibatch_size"] *= minibatch_size_multiplier(
        params["use_gpu"], params["use_all_avail_gpus"]
    )

    rl_parameters = RLParameters(**params["rl"])
    training_parameters = TrainingParameters(**params["training"])
    rainbow_parameters = RainbowDQNParameters(**params["rainbow"])

    trainer_params = ContinuousActionModelParameters(
        rl=rl_parameters, training=training_parameters, rainbow=rainbow_parameters
    )

    dataset = JSONDataset(
        params["training_data_path"], batch_size=training_parameters.minibatch_size
    )
    eval_dataset = JSONDataset(params["eval_data_path"], batch_size=16)
    state_normalization = read_norm_file(params["state_norm_data_path"])
    action_normalization = read_norm_file(params["action_norm_data_path"])

    num_batches = int(len(dataset) / training_parameters.minibatch_size)
    logger.info(
        "Read in batch data set {} of size {} examples. Data split "
        "into {} batches of size {}.".format(
            params["training_data_path"],
            len(dataset),
            num_batches,
            training_parameters.minibatch_size,
        )
    )

    trainer = ParametricDQNTrainer(
        trainer_params,
        state_normalization,
        action_normalization,
        use_gpu=params["use_gpu"],
        use_all_avail_gpus=params["use_all_avail_gpus"],
    )
    trainer = update_model_for_warm_start(trainer)
    state_preprocessor = Preprocessor(state_normalization, False)
    action_preprocessor = Preprocessor(action_normalization, False)

    evaluator = Evaluator(
        None,
        trainer_params.rl.gamma,
        trainer,
        metrics_to_score=trainer.metrics_to_score,
    )

    start_time = time.time()
    for epoch in range(params["epochs"]):
        dataset.reset_iterator()
        for batch_idx in range(num_batches):
            report_training_status(batch_idx, num_batches, epoch, params["epochs"])
            batch = dataset.read_batch(batch_idx)
            tdp = preprocess_batch_for_training(
                state_preprocessor, batch, action_preprocessor=action_preprocessor
            )

            tdp.set_type(trainer.dtype)
            trainer.train(tdp)

        eval_dataset.reset_iterator()
        accumulated_edp = None
        while True:
            batch = eval_dataset.read_batch(batch_idx)
            if batch is None:
                break
            tdp = preprocess_batch_for_training(
                state_preprocessor, batch, action_preprocessor=action_preprocessor
            )
            edp = EvaluationDataPage.create_from_tdp(tdp, trainer)
            if accumulated_edp is None:
                accumulated_edp = edp
            else:
                accumulated_edp = accumulated_edp.append(edp)
        accumulated_edp = accumulated_edp.compute_values(trainer.gamma)

        cpe_start_time = time.time()
        details = evaluator.evaluate_post_training(accumulated_edp)
        details.log()
        logger.info(
            "CPE evaluation took {} seconds.".format(time.time() - cpe_start_time)
        )

    through_put = (len(dataset) * params["epochs"]) / (time.time() - start_time)
    logger.info(
        "Training finished. Processed ~{} examples / s.".format(round(through_put))
    )

    return export_trainer_and_predictor(trainer, params["model_output_path"])


if __name__ == "__main__":
    params = parse_args(sys.argv)
    train_network(params)
