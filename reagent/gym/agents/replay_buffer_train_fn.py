#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional

from reagent.gym.types import ReplayBufferTrainFn, TrainerPreprocessor
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.training.trainer import Trainer


def replay_buffer_train_fn(
    trainer: Trainer,
    trainer_preprocessor: TrainerPreprocessor,
    training_freq: int,
    batch_size: int,
    replay_burnin: Optional[int] = None,
) -> ReplayBufferTrainFn:
    """ Called in post_step of agent to train based on replay buffer (RB).
        Args:
            trainer: responsible for having a .train method to train the model
            trainer_preprocessor: format RB output for trainer.train
            training_freq: how many steps in between trains
            batch_size: how big of a batch to sample
            replay_burnin: optional requirement for minimum size of RB before
                training begins. (i.e. burn in this many frames)
    """
    _num_steps = 0
    size_req = batch_size
    if replay_burnin is not None:
        size_req = max(size_req, replay_burnin)

    def train(replay_buffer: ReplayBuffer) -> None:
        """ To be called in post step """
        nonlocal _num_steps, size_req
        if replay_buffer.size >= size_req and _num_steps % training_freq == 0:
            train_batch = replay_buffer.sample_transition_batch(batch_size=batch_size)
            preprocessed_batch = trainer_preprocessor(train_batch)
            trainer.train(preprocessed_batch)
        _num_steps += 1
        return

    return train
