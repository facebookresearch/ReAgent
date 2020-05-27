#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import List

import gym
import numpy as np
import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.gym.policies.policy import Policy
from reagent.parameters import CONTINUOUS_TRAINING_ACTION_RANGE


def make_random_policy_for_env(env: gym.Env):
    if isinstance(env.action_space, gym.spaces.Discrete):
        # discrete action space
        return DiscreteRandomPolicy.create_for_env(env)
    elif isinstance(env.action_space, gym.spaces.Box):
        # continuous action space
        return ContinuousRandomPolicy.create_for_env(env)
    elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
        return MultiDiscreteRandomPolicy.create_for_env(env)
    else:
        raise NotImplementedError(f"{env.action_space} not supported")


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

    def act(self, obs: rlt.FeatureData) -> rlt.ActorOutput:
        """ Act randomly regardless of the observation. """
        obs: torch.Tensor = obs.float_features
        assert obs.dim() >= 2, f"obs has shape {obs.shape} (dim < 2)"
        batch_size = obs.shape[0]
        weights = torch.ones((batch_size, self.num_actions))

        # sample a random action
        m = torch.distributions.Categorical(weights)
        raw_action = m.sample()
        action = F.one_hot(raw_action, self.num_actions)
        log_prob = m.log_prob(raw_action).float()
        return rlt.ActorOutput(action=action, log_prob=log_prob)


class MultiDiscreteRandomPolicy(Policy):
    def __init__(self, num_action_vec: List[int]):
        self.num_action_vec = num_action_vec
        self.dists = [
            torch.distributions.Categorical(torch.ones(n) / n)
            for n in self.num_action_vec
        ]

    @classmethod
    def create_for_env(cls, env: gym.Env):
        action_space = env.action_space
        if not isinstance(action_space, gym.spaces.MultiDiscrete):
            raise ValueError(f"Invalid action space: {action_space}")

        return cls(action_space.nvec.tolist())

    def act(self, obs: rlt.FeatureData) -> rlt.ActorOutput:
        obs: torch.Tensor = obs.float_features
        batch_size, _ = obs.shape

        actions = []
        log_probs = []
        for m in self.dists:
            actions.append(m.sample((batch_size, 1)))
            log_probs.append(m.log_prob(actions[-1]).float())

        return rlt.ActorOutput(
            action=torch.cat(actions, dim=1),
            log_prob=torch.cat(log_probs, dim=1).sum(1, keepdim=True),
        )


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
            assert (
                len(action_space.shape) == 1
            ), f"Box with shape {action_space.shape} not supported."
            low, high = CONTINUOUS_TRAINING_ACTION_RANGE
            # broadcast low and high to shape
            np_ones = np.ones(action_space.shape)
            return cls(
                low=torch.tensor(low * np_ones), high=torch.tensor(high * np_ones)
            )
        else:
            raise NotImplementedError(f"action_space is {type(action_space)}")

    def act(self, obs: rlt.FeatureData) -> rlt.ActorOutput:
        """ Act randomly regardless of the observation. """
        obs: torch.Tensor = obs.float_features
        assert obs.dim() >= 2, f"obs has shape {obs.shape} (dim < 2)"
        batch_size = obs.size(0)
        # pyre-fixme[6]: Expected `Union[torch.Size, torch.Tensor]` for 1st param
        #  but got `Tuple[int]`.
        action = self.dist.sample((batch_size,))
        # sum over action_dim (since assuming i.i.d. per coordinate)
        log_prob = self.dist.log_prob(action).sum(1)
        return rlt.ActorOutput(action=action, log_prob=log_prob)
