#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
The agent can observe a character at one time. But the
reward is given based on last n (n>1) steps' observation (a string).
In this environment, the agent can observe a character ("A", "B") at
each time step, but the reward it receives actually depends on past 3 steps:
if the agent observes "ABB" in the past 3 steps, it receives +5 reward; if the
agent observes "BBB", it receives -5 reward; otherwise, the agent receives 0.
The action is the next character the agent wants to reveal, and the next state
is exactly the action just taken (i.e., the transition function only depends on
the action). Each episode is limited to 6 steps. Therefore, the optimal policy
is to choose actions "ABBABB" in sequence which results to +10 reward.
"""
import itertools
import logging
from collections import deque

import numpy as np
import torch

# pyre-fixme[21]: Could not find module `gym`.
from gym import Env

# pyre-fixme[21]: Could not find module `gym.spaces`.
from gym.spaces import Box, Discrete


logger = logging.getLogger(__name__)


MAX_STEP = 6
CHARACTERS = ["A", "B"]
STATE_DIM = ACTION_DIM = len(CHARACTERS)
SEQ_LEN = 3


# pyre-fixme[11]: Annotation `Env` is not defined as a type.
class StringGameEnv(Env):
    def __init__(self, max_steps=MAX_STEP):
        np.random.seed(123)
        torch.manual_seed(123)
        self.max_steps = max_steps
        self.reward_map = {}
        self._init_reward()
        logger.debug(self.reward_map)
        self.recent_actions = deque([], maxlen=SEQ_LEN)
        self.recent_states = deque([], maxlen=SEQ_LEN)
        self.cur_state = None
        self.action_space = Discrete(ACTION_DIM)
        self.observation_space = Box(low=0, high=1, shape=(STATE_DIM,))
        self.step_cnt = 0
        self.reset()

    def _init_reward(self):
        for seq_len in range(1, SEQ_LEN + 1):
            for k in itertools.product(CHARACTERS, repeat=seq_len):
                self.reward_map["".join(k)] = 0
        self.reward_map["ABB"] = 5
        self.reward_map["BBB"] = -5

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
        reward = self.reward_map[string]
        return reward, string

    def step(self, action):
        assert self.action_space.contains(action)
        assert self.done is False
        self.step_cnt += 1

        self.recent_states.append(self.cur_state)
        self.recent_actions.append(action)
        reward, info = self.get_reward()
        if self.step_cnt >= self.max_steps:
            self.done = True
        ob = self.get_observation()
        self.cur_state = ob

        return ob, reward, self.done, {"reward_str": info}

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
        self.recent_states = deque([], maxlen=SEQ_LEN)
        self.recent_actions = deque([], maxlen=SEQ_LEN)
        self.step_cnt = 0
        ob = self.get_observation()
        self.cur_state = ob
        return ob

    def print_internal_state(self):
        print("Step", self.step_cnt)

        def state_to_chr(s):
            state_index = np.nonzero(s)[0]
            if len(state_index) != 1:
                # initial state
                return "I"
            return CHARACTERS[state_index.item()]

        state_str = "".join([state_to_chr(s) for s in self.recent_states])
        action_str = "".join([CHARACTERS[c] for c in self.recent_actions])
        print(
            "Internal state: recent states {}, recent actions {}".format(
                state_str, action_str
            )
        )

    @staticmethod
    def print_ob(ob):
        return str(ob)

    @staticmethod
    def print_action(action):
        return CHARACTERS[action]
