#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import sys
import time

from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.thrift.core.ttypes import (
    DDPGModelParameters,
    DDPGNetworkParameters,
    DDPGTrainingParameters,
    RLParameters,
)
from ml.rl.training.ddpg_trainer import DDPGTrainer, construct_action_scale_tensor
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

def train_network(params):
    logger.info("Running Parametric DQN workflow with params:")
    logger.info(params)

    # Set minibatch size based on # of devices being used to train
    params["shared_training"]["minibatch_size"] *= minibatch_size_multiplier(
        params["use_gpu"], params["use_all_avail_gpus"]
    )

    rl_parameters = RLParameters(**params["rl"])
    training_parameters = DDPGTrainingParameters(**params["shared_training"])
    actor_parameters = DDPGNetworkParameters(**params["actor_training"])
    critic_parameters = DDPGNetworkParameters(**params["critic_training"])

    trainer_params = DDPGModelParameters(
        rl=rl_parameters,
        shared_training=training_parameters,
        actor_training=actor_parameters,
        critic_training=critic_parameters,
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

    min_action_range_tensor_serving, max_action_range_tensor_serving = construct_action_scale_tensor(
        action_normalization, trainer_params.action_rescale_map
    )

    trainer = DDPGTrainer(
        trainer_params,
        state_normalization,
        action_normalization,
        min_action_range_tensor_serving,
        max_action_range_tensor_serving,
        use_gpu=params["use_gpu"],
        use_all_avail_gpus=params["use_all_avail_gpus"],
    )
    trainer = update_model_for_warm_start(trainer)
    state_preprocessor = Preprocessor(state_normalization, False)
    action_preprocessor = Preprocessor(action_normalization, False)

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

    through_put = (len(dataset) * params["epochs"]) / (time.time() - start_time)
    logger.info(
        "Training finished. Processed ~{} examples / s.".format(round(through_put))
    )

    return export_trainer_and_predictor(trainer, params["model_output_path"])


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    params = parse_args(sys.argv)
    train_network(params)
