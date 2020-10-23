#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

"""
Traditional MAB setup has sequence length = 1 always. In this setup, the
distributions of the arms rewards changes every round, and the agent is presented
with some information and control about how the arms will change.
In particular, the observation includes "mu_changes", which is the possible changes
to mu; only the arm picked by agent will have it's mu_changes reflected.
This way, the next state depend on (only) the previous state and action;
hence this a MDP.

The reward for picking an action is the change in mu corresponding to that arm.
With following set-up, optimal policy can accumulate a reward of 500 per run.
Note that if the policy picks an illegal action at any time, its reward is upper
bounded by -500.
"""
import random

# pyre-fixme[21]: Could not find module `gym`.
import gym
import numpy as np
import reagent.types as rlt
import torch
from reagent.core.dataclasses import dataclass
from reagent.gym.envs.env_wrapper import EnvWrapper
from reagent.parameters import NormalizationData, NormalizationKey
from reagent.test.base.utils import only_continuous_normalizer


MAX_STEPS = 100
ABS_LOW = -1000.0
ABS_HIGH = 1000.0

MU_LOW = 0.0
MU_HIGH = 1000.0


def get_initial_mus():
    return torch.tensor([100.0] * 5)


def get_mu_changes():
    return torch.tensor([-10.0] * 5)


def get_legal_indices_mask():
    LEGAL_PROBS = torch.tensor([0.95, 1.0, 0.95, 0.8, 0.8])
    return torch.bernoulli(LEGAL_PROBS)


# illegal move causes game to end with a big BOOM!!!
INVALID_MOVE_PENALTY = -1000.0
IDLE_PENALTY = -25.0

NUM_ARMS = 5

# in the real world, IDs are not indices into embedding table
# thus, we offset vals to test hashing mechanism
ID_LIST_OFFSET = 1000000
ID_SCORE_LIST_OFFSET = 1500000


def clamp(x, lo, hi):
    return max(min(x, hi), lo)


@dataclass
class ChangingArms(EnvWrapper):
    num_arms: int = NUM_ARMS

    # pyre-fixme[11]: Annotation `Env` is not defined as a type.
    def make(self) -> gym.Env:
        return ChangingArmsEnv(self.num_arms)

    def _split_state(self, obs: np.ndarray):
        assert obs.shape == (3, self.num_arms), f"{obs.shape}."
        dense_val = torch.tensor(obs[0, :]).view(1, self.num_arms)
        id_list_val = torch.tensor(obs[1, :]).nonzero(as_tuple=True)[0].to(torch.long)
        id_score_list_val = torch.tensor(obs[2, :])
        return dense_val, id_list_val, id_score_list_val

    def obs_preprocessor(self, obs: np.ndarray) -> rlt.FeatureData:
        dense_val, id_list_val, id_score_list_val = self._split_state(obs)
        return rlt.FeatureData(
            # dense value
            float_features=dense_val,
            # (offset, value)
            id_list_features={
                "legal": (torch.tensor([0], dtype=torch.long), id_list_val)
            },
            # (offset, key, value)
            id_score_list_features={
                "mu_changes": (
                    torch.tensor([0], dtype=torch.long),
                    torch.arange(self.num_arms, dtype=torch.long),
                    id_score_list_val,
                )
            },
        )

    def serving_obs_preprocessor(self, obs: np.ndarray) -> rlt.ServingFeatureData:
        dense_val, id_list_val, id_score_list_val = self._split_state(obs)
        return rlt.ServingFeatureData(
            float_features_with_presence=(
                dense_val,
                torch.ones_like(dense_val, dtype=torch.uint8),
            ),
            id_list_features={
                100: (torch.tensor([0], dtype=torch.long), id_list_val + ID_LIST_OFFSET)
            },
            id_score_list_features={
                1000: (
                    torch.tensor([0], dtype=torch.long),
                    torch.arange(self.num_arms, dtype=torch.long)
                    + ID_SCORE_LIST_OFFSET,
                    id_score_list_val,
                )
            },
        )

    def split_state_transform(self, elem: torch.Tensor):
        """ For generate data """
        dense_val, id_list_val, id_score_list_val = self._split_state(elem.numpy())
        return (
            {i: s.item() for i, s in enumerate(dense_val.view(-1))},
            {100: (id_list_val + ID_LIST_OFFSET).tolist()},
            {
                1000: {
                    i + ID_SCORE_LIST_OFFSET: s.item()
                    for i, s in enumerate(id_score_list_val)
                }
            },
        )

    @property
    def normalization_data(self):
        return {
            NormalizationKey.STATE: NormalizationData(
                dense_normalization_parameters=only_continuous_normalizer(
                    list(range(self.num_arms)), MU_LOW, MU_HIGH
                )
            )
        }

    def trainer_preprocessor(self, obs: torch.Tensor):
        batch_size = obs.shape[0]
        assert obs.shape == (batch_size, 3, self.num_arms), f"{obs.shape}"
        dense_val = obs[:, 0, :].view(batch_size, self.num_arms)
        # extract one-hot encoded values from id_list
        batch_indices, id_list_val = obs[:, 1, :].nonzero(as_tuple=True)
        offsets = []
        prev_batch_idx = -1
        for i, batch_idx in enumerate(batch_indices.tolist()):
            if batch_idx > prev_batch_idx:
                offsets.extend([i] * (batch_idx - prev_batch_idx))
                prev_batch_idx = batch_idx
            else:
                assert batch_idx == prev_batch_idx
        # handle the case of trailing empty batches
        if batch_idx < batch_size - 1:
            offsets.extend([i] * (batch_size - 1 - batch_idx))
        assert len(offsets) == batch_size, f"{len(offsets)} != {batch_size}."
        id_list_offsets = torch.tensor(offsets)

        # id_score_list is easier because not one-hot encoded
        id_score_list_offsets = torch.tensor(
            list(range(0, batch_size * self.num_arms, self.num_arms))
        )
        id_score_list_keys = torch.arange(self.num_arms).repeat(batch_size)
        id_score_list_vals = obs[:, 2, :].reshape(-1)
        return rlt.FeatureData(
            # dense value
            float_features=dense_val,
            # (offset, value)
            id_list_features={"legal": (id_list_offsets, id_list_val)},
            # (offset, key, value)
            id_score_list_features={
                "mu_changes": (
                    id_score_list_offsets,
                    id_score_list_keys,
                    id_score_list_vals,
                )
            },
        )


class ChangingArmsEnv(gym.Env):
    """ This is just the gym environment, without extra functionality """

    def __init__(self, num_arms):
        self.seed(0)
        self.num_arms = num_arms
        self.max_steps = MAX_STEPS

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()
        assert (
            0 <= action and action <= self.num_arms
        ), f"out-of-bounds action {action}."
        reached_max_steps = self.num_steps >= self.max_steps
        self.num_steps += 1

        # idle action
        if action == self.num_arms:
            # simply return new state, without updating distributions
            # this is ideal when there aren't any legal actions, this
            # would generate a new batch of legal actions
            return self.state, IDLE_PENALTY, reached_max_steps, None

        # illegal action
        if action not in self.legal_indices:
            return self.state, INVALID_MOVE_PENALTY, True, None

        # update states for only the action selected
        prev = self.mus[action].item()
        self.mus[action] = clamp(prev + self.mu_changes[action], MU_LOW, MU_HIGH)
        reward = prev - self.mus[action].item()
        return self.state, reward, reached_max_steps, None

    def seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)

    def reset(self):
        # initialize the distributions
        self.num_steps = 0
        self.mus = get_initial_mus()
        return self.state

    @property
    def state(self):
        """
        State comprises of:
        - initial mus
        - legal_indices mask
        - randomly-generated mu changes
        """
        self.mu_changes = get_mu_changes()
        legal_indices_mask = get_legal_indices_mask()
        self.legal_indices = legal_indices_mask.nonzero(as_tuple=True)[0]
        result = torch.stack([self.mus, legal_indices_mask, self.mu_changes])
        return result.numpy()

    @property
    def observation_space(self):
        """
        It should really be a Dict, but we return them all stacked since it's
        more convenient for RB.
        """
        return gym.spaces.Box(ABS_LOW, ABS_HIGH, shape=(3, self.num_arms))

    @property
    def action_space(self):
        # Selecting 0,1,2...,num_arms-1 is selecting an arm.
        # If action is invalid, agent incurs a penalty.
        # If action is valid, action is an idx i, and reward
        # is a sample from ith distribution. At the same time
        # the ith distribution is updated with the changes.
        # Alternatively, can choose NULL (i.e. do-nothing) action
        # if action = num_arms
        return gym.spaces.Discrete(self.num_arms + 1)
