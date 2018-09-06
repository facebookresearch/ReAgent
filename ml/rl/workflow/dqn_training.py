#!/usr/bin/env python3

import logging
import sys
import time

import numpy as np
from ml.rl.thrift.core.ttypes import (
    DiscreteActionModelParameters,
    RainbowDQNParameters,
    RLParameters,
    TrainingParameters,
)
from ml.rl.training.dqn_trainer import DQNTrainer
from ml.rl.workflow.helpers import (
    export_trainer_and_predictor,
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

    trainer = DQNTrainer(trainer_params, state_normalization, params["use_gpu"])
    trainer = update_model_for_warm_start(trainer)

    start_time = time.time()
    for epoch in range(params["epochs"]):
        for batch_idx in range(num_batches):
            report_training_status(batch_idx, num_batches, epoch, params["epochs"])
            batch = dataset.read_batch(batch_idx)
            tdp = preprocess_batch_for_training(
                action_names, batch, state_normalization
            )
            trainer.train(tdp)

    through_put = (len(dataset) * params["epochs"]) / (time.time() - start_time)
    logger.info(
        "Training finished. Processed ~{} examples / s.".format(round(through_put))
    )
    export_trainer_and_predictor(trainer, params["model_output_path"])


if __name__ == "__main__":
    params = parse_args(sys.argv)
    train_network(params)
