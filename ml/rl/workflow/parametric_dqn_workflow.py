#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import sys
import time

import numpy as np
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.thrift_handler import (
    ContinuousActionModelParameters,
    InTrainingCPEParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.training.evaluator import Evaluator
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
    if params["in_training_cpe"] is not None:
        in_training_cpe_parameters = InTrainingCPEParameters(
            **params["in_training_cpe"]
        )
    else:
        in_training_cpe_parameters = None

    trainer_params = ContinuousActionModelParameters(
        rl=rl_parameters,
        training=training_parameters,
        rainbow=rainbow_parameters,
        in_training_cpe=in_training_cpe_parameters,
    )

    dataset = JSONDataset(
        params["training_data_path"], batch_size=training_parameters.minibatch_size
    )
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
    state_preprocessor = Preprocessor(state_normalization, params["use_gpu"])
    action_preprocessor = Preprocessor(action_normalization, params["use_gpu"])

    if trainer_params.in_training_cpe is not None:
        evaluator = Evaluator(
            None,
            100,
            trainer_params.rl.gamma,
            trainer,
            trainer_params.in_training_cpe.mdp_sampled_rate,
        )
    else:
        evaluator = Evaluator(
            None,
            100,
            trainer_params.rl.gamma,
            trainer,
            float(DEFAULT_NUM_SAMPLES_FOR_CPE) / len(dataset),
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
            trainer.train(tdp, evaluator)

            evaluator.collect_parametric_action_samples(
                mdp_ids=tdp.mdp_ids,
                sequence_numbers=tdp.sequence_numbers.cpu().numpy(),
                logged_state_actions=np.concatenate(
                    (tdp.states.cpu().numpy(), tdp.actions.cpu().numpy()), axis=1
                ),
                logged_rewards=tdp.rewards.cpu().numpy(),
                logged_propensities=tdp.propensities.cpu().numpy(),
                logged_terminals=(1.0 - tdp.not_terminals),
                possible_state_actions=tdp.state_pas_concat.cpu().numpy(),
                pas_lens=tdp.possible_actions_lengths.cpu().numpy(),
            )

        cpe_start_time = time.time()
        evaluator.recover_samples_to_be_unshuffled()
        evaluator.score_cpe(trainer_params.rl.gamma)
        evaluator.clear_collected_samples()
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
