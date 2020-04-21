#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import Any, Optional

import reagent.types as rlt
import torch
from reagent.gym.types import ActionPreprocessor, PostStep, TrainerPreprocessor
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.training.rl_dataset import RLDataset
from reagent.training.trainer import Trainer


def train_with_replay_buffer_post_step(
    replay_buffer: ReplayBuffer,
    trainer: Trainer,
    trainer_preprocessor: TrainerPreprocessor,
    training_freq: int,
    batch_size: int,
    replay_burnin: Optional[int] = None,
) -> PostStep:
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

    def post_step(
        obs: Any,
        actor_output: rlt.ActorOutput,
        reward: float,
        terminal: bool,
        possible_actions_mask: Optional[torch.Tensor],
    ) -> None:
        nonlocal _num_steps

        action = actor_output.action.numpy()
        log_prob = actor_output.log_prob.numpy()
        if possible_actions_mask is None:
            possible_actions_mask = torch.ones_like(actor_output.action).to(torch.bool)
        possible_actions_mask = possible_actions_mask.numpy()
        replay_buffer.add(
            obs, action, reward, terminal, possible_actions_mask, log_prob
        )

        if replay_buffer.size >= size_req and _num_steps % training_freq == 0:
            train_batch = replay_buffer.sample_transition_batch(batch_size=batch_size)
            preprocessed_batch = trainer_preprocessor(train_batch)
            trainer.train(preprocessed_batch)
        _num_steps += 1
        return

    return post_step


# TODO: do not poass action_preprocessor here
def log_data_post_step(
    dataset: RLDataset, action_preprocessor: ActionPreprocessor, mdp_id: str
) -> PostStep:
    sequence_number = 0

    def post_step(
        obs: Any,
        actor_output: rlt.ActorOutput,
        reward: float,
        terminal: bool,
        possible_actions_mask: Optional[torch.Tensor],
    ) -> None:
        """ log data into dataset """
        nonlocal sequence_number

        if possible_actions_mask is None:
            possible_actions_mask = torch.ones_like(actor_output.action).to(torch.bool)

        if terminal:
            possible_actions_mask = torch.zeros_like(actor_output.action).to(torch.bool)

        # timeline operator expects str for disc and map<str, double> for cts
        # TODO: case for cts
        action = str(action_preprocessor(actor_output))

        dataset.insert_pre_timeline_format(
            mdp_id=None,
            sequence_number=sequence_number,
            state=obs,
            action=action,
            reward=reward,
            possible_actions=None,
            time_diff=1,
            action_probability=actor_output.log_prob.exp().item(),
            possible_actions_mask=possible_actions_mask,
        )
        sequence_number += 1
        return

    return post_step
