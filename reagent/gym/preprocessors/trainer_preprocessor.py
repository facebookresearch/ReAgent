#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

""" Get default preprocessors for training time. """

import inspect
import logging

import torch
from reagent.training.trainer import Trainer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def make_replay_buffer_trainer_preprocessor(trainer: Trainer, device: torch.device):
    sig = inspect.signature(trainer.train)
    logger.info(f"Deriving trainer_preprocessor from {sig.parameters}")
    # Assuming training_batch is in the first position (excluding self)
    assert (
        list(sig.parameters.keys())[0] == "training_batch"
    ), f"{sig.parameters} doesn't have training batch in first position."
    training_batch_type = sig.parameters["training_batch"].annotation
    assert training_batch_type != inspect.Parameter.empty
    if not hasattr(training_batch_type, "from_replay_buffer"):
        raise NotImplementedError(
            f"{training_batch_type} does not implement from_replay_buffer"
        )

    def trainer_preprocessor(batch):
        retval = training_batch_type.from_replay_buffer(batch)
        return retval.to(device)

    return trainer_preprocessor
