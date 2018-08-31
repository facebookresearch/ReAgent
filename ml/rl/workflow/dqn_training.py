#!/usr/bin/env python3

import logging
import sys

import numpy as np
from ml.rl.thrift.core.ttypes import (
    DiscreteActionModelParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.training.dqn_trainer import DQNTrainer
from ml.rl.workflow import helpers
from ml.rl.workflow.training_data_reader import (
    JSONDataset,
    preprocess_batch_for_training,
    read_norm_params,
)


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def train_network(params):
    logger.info("Running DQN workflow with params:")
    logger.info(params)

    action_names = np.array(params["actions"])
    rl_parameters = RLParameters(**params["rl"])
    training_parameters = TrainingParameters(**params["training"])
    rainbow_parameters = RainbowDQNParameters(**params["rainbow"])

    trainer_params = DiscreteActionModelParameters(
        actions=params["actions"],
        rl=rl_parameters,
        training=training_parameters,
        rainbow=rainbow_parameters,
    )

    dataset = JSONDataset(
        params["training_data_path"], batch_size=training_parameters.minibatch_size
    )
    norm_data = JSONDataset(params["state_norm_data_path"])
    state_normalization = read_norm_params(norm_data.read_all())

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

    trainer = DQNTrainer(trainer_params, state_normalization, params["use_gpu"])

    for epoch in range(params["epochs"]):
        for batch_idx in range(num_batches):
            helpers.report_training_status(
                batch_idx, num_batches, epoch, params["epochs"]
            )
            batch = dataset.read_batch(batch_idx)
            tdp = preprocess_batch_for_training(
                action_names, batch, state_normalization
            )
            trainer.train(tdp)

    logger.info(
        "Training finished. Saving PyTorch model to {}".format(
            params["pytorch_output_path"]
        )
    )
    helpers.save_model_to_file(trainer, params["pytorch_output_path"])


if __name__ == "__main__":
    params = helpers.parse_args(sys.argv)
    train_network(params)
