#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Optional

import torch
from reagent.gym.policies.policy import Policy
from reagent.gym.types import ActionPreprocessor, ReplayBufferAddFn, ReplayBufferTrainFn
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer


class Agent:
    def __init__(
        self,
        policy: Policy,
        action_preprocessor: ActionPreprocessor,
        replay_buffer: Optional[ReplayBuffer] = None,
        replay_buffer_add_fn: Optional[ReplayBufferAddFn] = None,
        replay_buffer_train_fn: Optional[ReplayBufferTrainFn] = None,
    ):
        """
        The Agent orchestrates the interactions on our RL components, given
        the interactions with the environment.

        Args:
            policy: Policy that acts given preprocessed input
            replay_buffer: if provided, inserts each experience via the
                replay_buffer_add_fn
            replay_buffer_add_fn: fn of the form
                (replay_buffer, obs, action, r, t) -> void
                which adds an experience into the given replay buffer
            replay_buffer_train_fn: called in poststep after adding experience.
                Performs training steps based on replay buffer samples.
        """
        self.policy = policy
        self.action_preprocessor = action_preprocessor
        self.replay_buffer = replay_buffer
        self.replay_buffer_add_fn = replay_buffer_add_fn
        self.replay_buffer_train_fn = replay_buffer_train_fn

        # intermediate state between act and post_step
        self._obs: Any = None
        self._actor_output: Any = None
        self._possible_actions_mask = None

    def act(
        self, obs: Any, possible_actions_mask: Optional[torch.Tensor] = None
    ) -> Any:
        actor_output = self.policy.act(obs, possible_actions_mask)
        if self.replay_buffer:
            self._obs = obs
            self._actor_output = actor_output
            self._possible_actions_mask = possible_actions_mask
        action_for_env = self.action_preprocessor(actor_output)
        return action_for_env

    def post_step(self, reward: float, terminal: bool, *args):
        """ to be called after step(action) """
        if self.replay_buffer:
            assert self._obs is not None
            assert self._actor_output is not None
            assert self.replay_buffer_add_fn is not None
            assert self.replay_buffer_train_fn is not None
            self.replay_buffer_add_fn(
                self.replay_buffer,
                self._obs,
                self._actor_output,
                reward,
                terminal,
                self._possible_actions_mask,
            )
            self._obs = None
            self._actor_output = None
            self._possible_actions_mask = None

            self.replay_buffer_train_fn(self.replay_buffer)
