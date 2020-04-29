#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import inspect
import logging
from typing import Any, Optional, Union

import gym
import numpy as np
import reagent.types as rlt
import torch
from reagent.gym.types import PostStep
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.training.rl_dataset import RLDataset
from reagent.training.trainer import Trainer


logger = logging.getLogger(__name__)


def train_with_replay_buffer_post_step(
    replay_buffer: ReplayBuffer,
    trainer: Trainer,
    training_freq: int,
    batch_size: int,
    replay_burnin: Optional[int] = None,
    trainer_preprocessor=None,
    device: Optional[Union[str, torch.device]] = None,
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
    if device is not None and isinstance(device, str):
        device = torch.device(device)

    _num_steps = 0
    size_req = batch_size
    if replay_burnin is not None:
        size_req = max(size_req, replay_burnin)

    if trainer_preprocessor is None:
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
            if device is not None:
                retval = retval.to(device)
            return retval

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


def log_data_post_step(dataset: RLDataset, mdp_id: str, env: gym.Env) -> PostStep:
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

        # actor_output = actor_output.squeeze(0)
        if isinstance(env.action_space, gym.spaces.Discrete):
            # TimelineOperator expects str for discrete action
            action = str(actor_output.action.argmax().item())
            if possible_actions_mask is None:
                possible_actions_mask = torch.ones_like(actor_output.action).to(
                    torch.bool
                )

            if terminal:
                possible_actions_mask = torch.zeros_like(actor_output.action).to(
                    torch.bool
                )
        elif isinstance(env.action_space, gym.spaces.Box):
            # TimelineOperator expects map<long, double> for discrete action
            assert actor_output.action.dim() == 1, f"action dim > 1 in {actor_output}"
            action = {
                i: actor_output.action[i].item()
                for i in range(actor_output.action.size(0))
            }
        else:
            raise NotImplementedError(f"{env.action_space} not supported!")
        # TODO: make output of policy the desired type already (which means
        # altering RB logic to store scalar types) What to do about continuous?

        action_prob = actor_output.log_prob.exp().item()
        possible_actions = None  # TODO: this shouldn't be none if env passes it
        time_diff = 1  # TODO: should this be hardcoded?

        # Some environments return numpy instead
        if isinstance(reward, np.float64):
            reward = reward.item()

        dataset.insert_pre_timeline_format(
            mdp_id=mdp_id,
            sequence_number=sequence_number,
            state=obs,
            action=action,
            reward=reward,
            possible_actions=possible_actions,
            time_diff=time_diff,
            action_probability=action_prob,
            possible_actions_mask=possible_actions_mask,
        )
        sequence_number += 1
        return

    return post_step
