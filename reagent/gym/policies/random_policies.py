#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

from typing import List, Optional

import gym
import numpy as np
import reagent.core.types as rlt
import torch
import torch.nn.functional as F
from reagent.core.parameters import CONTINUOUS_TRAINING_ACTION_RANGE
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.scorers.discrete_scorer import apply_possible_actions_mask


def make_random_policy_for_env(env: gym.Env) -> Policy:
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
    def __init__(self, num_actions: int) -> None:
        """Random actor for accumulating random offline data."""
        self.num_actions = num_actions

    @classmethod
    def create_for_env(cls, env: gym.Env) -> "DiscreteRandomPolicy":
        action_space = env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            return cls(num_actions=action_space.n)
        elif isinstance(action_space, gym.spaces.Box):
            raise NotImplementedError(f"Try continuous random policy instead")
        else:
            raise NotImplementedError(f"action_space is {type(action_space)}")

    def act(
        self, obs: rlt.FeatureData, possible_actions_mask: Optional[torch.Tensor] = None
    ) -> rlt.ActorOutput:
        """Act randomly regardless of the observation."""
        # pyre-fixme[35]: Target cannot be annotated.
        obs: torch.Tensor = obs.float_features
        assert obs.dim() >= 2, f"obs has shape {obs.shape} (dim < 2)"
        assert obs.shape[0] == 1, f"obs has shape {obs.shape} (0th dim != 1)"
        batch_size = obs.shape[0]
        scores = torch.ones((batch_size, self.num_actions))
        scores = apply_possible_actions_mask(
            scores, possible_actions_mask, invalid_score=0.0
        )

        # sample a random action
        m = torch.distributions.Categorical(scores)
        raw_action = m.sample()
        action = F.one_hot(raw_action, self.num_actions)
        log_prob = m.log_prob(raw_action).float()
        return rlt.ActorOutput(action=action, log_prob=log_prob)


class MultiDiscreteRandomPolicy(Policy):
    def __init__(self, num_action_vec: List[int]) -> None:
        self.num_action_vec = num_action_vec
        self.dists = [
            torch.distributions.Categorical(torch.ones(n) / n)
            for n in self.num_action_vec
        ]

    @classmethod
    def create_for_env(cls, env: gym.Env) -> "MultiDiscreteRandomPolicy":
        action_space = env.action_space
        if not isinstance(action_space, gym.spaces.MultiDiscrete):
            raise ValueError(f"Invalid action space: {action_space}")

        return cls(action_space.nvec.tolist())

    # TODO: consider possible_actions_mask
    def act(
        self, obs: rlt.FeatureData, possible_actions_mask: Optional[torch.Tensor] = None
    ) -> rlt.ActorOutput:
        # pyre-fixme[35]: Target cannot be annotated.
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
    def __init__(self, low: torch.Tensor, high: torch.Tensor) -> None:
        self.low = low
        self.high = high
        assert low.shape == high.shape, (
            f"low.shape = {low.shape}, high.shape = {high.shape}"
        )
        self.dist = torch.distributions.uniform.Uniform(self.low, self.high)

    @classmethod
    def create_for_env(cls, env: gym.Env) -> "ContinuousRandomPolicy":
        action_space = env.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            raise NotImplementedError(
                f"Action space is discrete. Try using DiscreteRandomPolicy instead."
            )
        elif isinstance(action_space, gym.spaces.Box):
            assert len(action_space.shape) == 1, (
                f"Box with shape {action_space.shape} not supported."
            )
            low, high = CONTINUOUS_TRAINING_ACTION_RANGE
            # broadcast low and high to shape
            np_ones = np.ones(action_space.shape)
            return cls(
                low=torch.tensor(low * np_ones), high=torch.tensor(high * np_ones)
            )
        else:
            raise NotImplementedError(f"action_space is {type(action_space)}")

    def act(
        self, obs: rlt.FeatureData, possible_actions_mask: Optional[torch.Tensor] = None
    ) -> rlt.ActorOutput:
        """Act randomly regardless of the observation."""
        # pyre-fixme[35]: Target cannot be annotated.
        obs: torch.Tensor = obs.float_features
        assert obs.dim() >= 2, f"obs has shape {obs.shape} (dim < 2)"
        batch_size = obs.size(0)
        action = self.dist.sample((batch_size,))
        # sum over action_dim (since assuming i.i.d. per coordinate)
        log_prob = self.dist.log_prob(action).sum(1)
        return rlt.ActorOutput(action=action, log_prob=log_prob)
