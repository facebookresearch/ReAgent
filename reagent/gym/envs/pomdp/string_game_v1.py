#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
A game with a stochastic length of the MDP but no longer than 3.

An agent can choose one character to reveal (either "A" or "B") as the action,
and the next state is exactly the action just taken (i.e., the transition
function only depends on the action). Each episode is limited to 3 steps.

There is some probability to terminate at any step (but the agent must terminate if
making 3 steps)
If the current state is "A", the agent has 0.5 probability to make to the next step.
If the current state is "B", the agent has 0.9 probability to make to the next step.
The reward is given at the terminal state, based on the accumulated observation (a string).

If the agent observes "AAA" (survive the first 2 steps and terminate at the last step
 no matter what action taken), it receives +5 reward.
If the agent observes "BA" (survive the first step and terminate at the second step),
it receives +4 reward.
For all other scenarios, the agent receives 0 reward.

If we plan for 3 steps ahead from the beginning, "A" is the better action to take first.
If we plan with consideration of termination probabilities, "B" is better. Because:
The expected Q-value of "A" = 0.5 * 0 + 0.5 * max(0.5 * 0 + 0.5 * max(5, 0), 0) = 1.25
The expected Q-value of "B" = 0.1 * 0 + 0.9 * max(0.5 * 4 + 0.5 * max(0, 0), 0) = 1.8
"""
import logging
from collections import deque, defaultdict

import numpy as np
import torch
from gym import Env
from gym.spaces import Box, Discrete


logger = logging.getLogger(__name__)


MAX_STEP = 3
CHARACTERS = ["A", "B"]
STATE_DIM = ACTION_DIM = len(CHARACTERS)


class StringGameEnvV1(Env):
    def __init__(self, max_steps=MAX_STEP):
        np.random.seed(123)
        torch.manual_seed(123)
        self.max_steps = max_steps
        self.reward_map = defaultdict(float)
        self.terminal_probs = defaultdict(float)
        self._init_reward_and_terminal_probs()
        self.recent_actions = deque([], maxlen=MAX_STEP)
        self.action_space = Discrete(ACTION_DIM)
        self.observation_space = Box(low=0, high=1, shape=(STATE_DIM,))
        self.step_cnt = 0
        self.reset()

    def _init_reward_and_terminal_probs(self):
        self.reward_map["AAA"] = 5.0
        self.reward_map["BA"] = 4.0
        self.terminal_probs["A"] = 0.5
        self.terminal_probs["B"] = 0.1

    def seed(self, seed=None):
        np.random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def random_action():
        return np.random.randint(0, ACTION_DIM)

    def get_reward(self):
        """
        The function you can write to customize rewards. In this
        specific environment, the reward only depends on action history
        """
        recent_characters = [CHARACTERS[c] for c in list(self.recent_actions)]
        string = "".join(recent_characters)
        if not self.done:
            reward = 0
        else:
            reward = self.reward_map[string]
        return reward, string

    def step(self, action):
        assert self.action_space.contains(action)
        assert self.done is False

        self.step_cnt += 1
        self.recent_actions.append(action)
        if self.step_cnt >= self.max_steps:
            self.done = True
        else:
            self.done = self.sample_terminal(action)
        reward, info = self.get_reward()
        ob = self.get_observation()

        return ob, reward, self.done, {"reward_str": info}

    def sample_terminal(self, action):
        terminal_probability = self.terminal_probs[CHARACTERS[action]]
        if np.random.rand() < terminal_probability:
            return True
        return False

    def get_observation(self):
        """
        The function you can write to customize transitions. In this
        specific environment, the next state is exactly the latest action taken.
        The initial observation is all zeros.
        """
        ob = np.zeros(STATE_DIM)
        if len(self.recent_actions) > 0:
            ob[self.recent_actions[-1]] = 1
        return ob

    def reset(self):
        self.done = False
        self.recent_actions = deque([], maxlen=MAX_STEP)
        self.step_cnt = 0
        ob = self.get_observation()
        return ob

    def print_internal_state(self):
        action_str = "".join([CHARACTERS[c] for c in self.recent_actions])
        logger.debug(
            f"Step {self.step_cnt}, recent actions {action_str}, terminal={self.done}"
        )

    @staticmethod
    def print_ob(ob):
        return str(ob)

    @staticmethod
    def print_action(action):
        return CHARACTERS[action]
