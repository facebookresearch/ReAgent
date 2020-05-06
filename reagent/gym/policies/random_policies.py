#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Optional

import gym
import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.gym.policies.policy import Policy


"""
TODO: remove explicit argument of possible_actions_mask since cts doesn't have.
"""


class DiscreteRandomPolicy(Policy):
    def __init__(self, num_actions: int):
        """ Random actor for accumulating random offline data. """
        self.num_actions = num_actions

    @classmethod
    def create_for_env(cls, env: gym.Env):
        action_space = env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            return cls(num_actions=action_space.n)
        elif isinstance(action_space, gym.spaces.Box):
            raise NotImplementedError(f"Try continuous random policy instead")
        else:
            raise NotImplementedError(f"action_space is {type(action_space)}")

    def act(
        self,
        obs: rlt.PreprocessedState,
        possible_actions_mask: Optional[torch.Tensor] = None,
    ) -> rlt.ActorOutput:
        """ Act randomly regardless of the observation. """
        # pyre-fixme[9]: obs has type `PreprocessedState`; used as `Tensor`.
        obs = obs.state.float_features
        assert obs.dim() >= 2, f"obs has shape {obs.shape} (dim < 2)"
        batch_size = obs.shape[0]
        weights = torch.ones((batch_size, self.num_actions))
        if possible_actions_mask:
            assert possible_actions_mask.shape == (batch_size, self.num_actions)
            # element-wise multiplication
            weights = weights * possible_actions_mask

        # sample a random action
        m = torch.distributions.Categorical(weights)
        raw_action = m.sample()
        action = F.one_hot(raw_action, self.num_actions)
        log_prob = m.log_prob(raw_action).float()
        return rlt.ActorOutput(action=action, log_prob=log_prob)


class ContinuousRandomPolicy(Policy):
    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        self.low = low
        self.high = high
        assert (
            low.shape == high.shape
        ), f"low.shape = {low.shape}, high.shape = {high.shape}"
        self.dist = torch.distributions.uniform.Uniform(self.low, self.high)

    @classmethod
    def create_for_env(cls, env: gym.Env):
        action_space = env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            raise NotImplementedError(
                f"Action space is discrete. Try using DiscreteRandomPolicy instead."
            )
        elif isinstance(action_space, gym.spaces.Box):
            low = torch.tensor(action_space.low).float()
            high = torch.tensor(action_space.high).float()
            return cls(low=low, high=high)
        else:
            raise NotImplementedError(f"action_space is {type(action_space)}")

    # pyre-fixme[14]: `act` overrides method defined in `Policy` inconsistently.
    def act(self, obs: rlt.PreprocessedState) -> rlt.ActorOutput:
        """ Act randomly regardless of the observation. """
        # pyre-fixme[9]: obs has type `PreprocessedState`; used as `Tensor`.
        obs = obs.state.float_features
        assert obs.dim() >= 2, f"obs has shape {obs.shape} (dim < 2)"
        batch_size = obs.size(0)
        action = self.dist.sample((batch_size,))
        log_prob = self.dist.log_prob(action)
        return rlt.ActorOutput(action=action, log_prob=log_prob)
