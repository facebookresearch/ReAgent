#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import os
import sys
import time

import numpy as np
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.thrift.core.ttypes import (
    DiscreteActionModelParameters,
    InTrainingCPEParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.training.dqn_trainer import DQNTrainer
from ml.rl.training.evaluator import Evaluator
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
from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger.setLevel(logging.INFO)

DEFAULT_NUM_SAMPLES_FOR_CPE = 5000


def train_network(params):
    writer = None
    if params["model_output_path"] is not None:
        writer = SummaryWriter(log_dir=params["model_output_path"])

    logger.info("Running DQN workflow with params:")
    logger.info(params)

    # Set minibatch size based on # of devices being used to train
    params["training"]["minibatch_size"] *= minibatch_size_multiplier(
        params["use_gpu"], params["use_all_avail_gpus"]
    )

    action_names = np.array(params["actions"])
    rl_parameters = RLParameters(**params["rl"])
    training_parameters = TrainingParameters(**params["training"])
    rainbow_parameters = RainbowDQNParameters(**params["rainbow"])
    if params["in_training_cpe"] is not None:
        in_training_cpe_parameters = InTrainingCPEParameters(
            **params["in_training_cpe"]
        )
    else:
        in_training_cpe_parameters = None

    trainer_params = DiscreteActionModelParameters(
        actions=params["actions"],
        rl=rl_parameters,
        training=training_parameters,
        rainbow=rainbow_parameters,
        in_training_cpe=in_training_cpe_parameters,
    )

    dataset = JSONDataset(
        params["training_data_path"], batch_size=training_parameters.minibatch_size
    )
    state_normalization = read_norm_file(params["state_norm_data_path"])

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

    trainer = DQNTrainer(
        trainer_params,
        state_normalization,
        use_gpu=params["use_gpu"],
        use_all_avail_gpus=params["use_all_avail_gpus"],
    )
    trainer = update_model_for_warm_start(trainer)
    preprocessor = Preprocessor(state_normalization, params["use_gpu"])

    if trainer_params.in_training_cpe is not None:
        evaluator = Evaluator(
            trainer_params.actions,
            trainer_params.rl.gamma,
            trainer,
            trainer_params.in_training_cpe.mdp_sampled_rate,
            metrics_to_score=trainer.metrics_to_score,
        )
    else:
        evaluator = Evaluator(
            trainer_params.actions,
            trainer_params.rl.gamma,
            trainer,
            float(DEFAULT_NUM_SAMPLES_FOR_CPE) / len(dataset),
            metrics_to_score=trainer.metrics_to_score,
        )

    start_time = time.time()
    for epoch in range(int(params["epochs"])):
        dataset.reset_iterator()
        for batch_idx in range(num_batches):
            report_training_status(batch_idx, num_batches, epoch, int(params["epochs"]))
            batch = dataset.read_batch(batch_idx)
            tdp = preprocess_batch_for_training(preprocessor, batch, action_names)

            tdp.set_type(trainer.dtype)
            training_metadata = trainer.train(tdp)

            evaluator.collect_discrete_action_samples(
                mdp_ids=tdp.mdp_ids,
                sequence_numbers=tdp.sequence_numbers.cpu().numpy(),
                states=tdp.states.cpu().numpy(),
                logged_actions=tdp.actions.cpu().numpy(),
                logged_rewards=tdp.rewards.cpu().numpy(),
                logged_propensities=tdp.propensities.cpu().numpy(),
                logged_terminals=np.invert(
                    tdp.not_terminals.cpu().numpy().astype(np.bool)
                ),
                model_rewards=training_metadata["model_rewards"],
                metrics=tdp.rewards.cpu().numpy(),  # Dummy until metrics CPE ported to open source
            )

        cpe_start_time = time.time()
        evaluator.recover_samples_to_be_unshuffled()
        evaluator.score_cpe(trainer_params.rl.gamma)
        if writer is not None:
            evaluator.log_to_tensorboard(writer, epoch)
        evaluator.clear_collected_samples()
        logger.info(
            "CPE evaluation took {} seconds.".format(time.time() - cpe_start_time)
        )

    through_put = (len(dataset) * int(params["epochs"])) / (time.time() - start_time)
    logger.info(
        "Training finished. Processed ~{} examples / s.".format(round(through_put))
    )

    if writer is not None:
        writer.close()

    return export_trainer_and_predictor(trainer, params["model_output_path"])


if __name__ == "__main__":
    params = parse_args(sys.argv)
    train_network(params)
