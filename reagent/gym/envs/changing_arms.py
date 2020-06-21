#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

"""
Traditional MAB setup has sequence length = 1 always. In this setup, the
distributions of the arms rewards can change. The agent is presented with
the possible changes to mu and sigma and only the chosen arm will have the
changes reflected (the other arms remain unchanged). This way, the next state
depend on (only) the previous state and action; hence this a MDP.

In general, the should always be at least one arm that can increase, so
a theoretically best agent should get infinite rewards.
"""
import random

import gym
import numpy as np
import torch


INITIAL_MUS = torch.tensor([0.0, 5.0, 3.0, 7.0, 10.0])
MU_LOW = 0
MU_HIGH = 100000.0

# sample range
SAMPLE_LOW = -100000.0
SAMPLE_HIGH = 100000.0

# changes range=
MU_CHANGES_LOW = -2
MU_CHANGES_HIGH = 2

# each arm has this prob of being legal each round
# illegal move causes game to end with a big BOOM!!!
LEGAL_PROB = 0.8
INVALID_MOVE_PENALTY = -100
IDLE_PENALTY = -5

ABS_LOW = -100000.0
ABS_HIGH = 100000.0

NUM_ARMS = 5


def clamp(x, lo, hi):
    return max(min(x, hi), lo)


def get_mu_changes():
    # return torch.randint(
    #         MU_CHANGES_LOW,
    #         MU_CHANGES_HIGH + 1,
    #         size=(self.num_arms,),
    #         dtype=torch.float32,
    #     )
    return torch.tensor([0.2, 0.1, 0.01, -0.1, -0.2])


class ChangingArms(gym.Env):
    def __init__(self):
        self.seed(0)
        self.num_arms = NUM_ARMS
        self.legal_probs = torch.tensor([LEGAL_PROB] * self.num_arms)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()
        assert (
            0 <= action and action <= self.num_arms
        ), f"out-of-bounds action {action}."

        # idle action
        if action == self.num_arms:
            # simply return new state, without updating distributions
            # this is ideal when there aren't any legal actions, this
            # would generate a new batch of legal actions
            return self.state, IDLE_PENALTY, False, None

        # illegal action
        if action not in self.legal_indices:
            return self.state, INVALID_MOVE_PENALTY, True, None

        reward = self.mus[action].item()

        # update states for only the action selected
        self.mus[action] = clamp(
            self.mus[action] + self.mu_changes[action], MU_LOW, MU_HIGH
        )
        return self.state, reward, False, None

    def seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)

    def reset(self):
        # initialize the distributions
        self.mus = INITIAL_MUS.clone()
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
        legal_indices_mask = torch.bernoulli(self.legal_probs)
        self.legal_indices = legal_indices_mask.nonzero().squeeze(1)
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
