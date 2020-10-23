#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
Simple environment to test possible_actions_mask.
State simply tells you which iteration it is, but doesn't tell anything about
which action to take, so only source of info is possible_actions_mask.
The Q-value of each action to converge to the (discounted) value of the MDP.

The value of the MDP should be 10 * max_steps = 200
"""

# pyre-fixme[21]: Could not find module `gym`.
import gym
import numpy as np

# pyre-fixme[21]: Could not find module `gym.spaces`.
from gym.spaces import Box, Discrete


def _get_state(step_idx, max_steps):
    """ One-hot encoding of which state we're on """
    zeros = np.zeros(max_steps, dtype=np.float32)
    if step_idx == max_steps:
        return zeros
    assert 0 <= step_idx and step_idx < max_steps
    zeros[step_idx] = 1.0
    return zeros


# pyre-fixme[11]: Annotation `Env` is not defined as a type.
class PossibleActionsMaskTester(gym.Env):
    def __init__(self):
        self.max_steps = 20
        self.action_num = 4
        self.cur_step = -1
        self.observation_space = Box(0.0, 1.0, shape=(self.max_steps,))
        self.action_space = Discrete(n=self.action_num)

    def _update_possible_actions_mask(self):
        self.legal_action = np.random.randint(self.action_num)
        self.possible_actions_mask = np.zeros(self.action_num, dtype=np.bool)
        self.possible_actions_mask[self.legal_action] = True

    def _get_state(self):
        return _get_state(self.cur_step, self.max_steps)

    def reset(self):
        self.cur_step = 0
        self._update_possible_actions_mask()
        return self._get_state()

    def step(self, action):
        reward = 10.0 if action == self.legal_action else 0.0
        terminal = self.cur_step == (self.max_steps - 1)
        self.cur_step += 1
        self._update_possible_actions_mask()
        return self._get_state(), reward, terminal, None
