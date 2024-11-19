#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
"""
A simple linear dynamic system
https://www.argmin.net/2018/02/08/lqr/
"""

import logging

import numpy as np
from gym import Env
from gym.spaces import Box


logger = logging.getLogger(__name__)


class LinDynaEnv(Env):
    """
    A linear dynamical system characterized by A, B, Q, and R.

    Suppose x_t is current state, u_t is current action, then:

    x_t+1 = A x_t + B u_t
    Reward_t = x_t' Q x_t + u_t' R u_t
    """

    def __init__(self):
        self.max_steps = 4
        self.state_dim = 3
        self.action_dim = 2
        self.action_space = Box(low=-3, high=3, shape=(self.action_dim,))
        self.observation_space = Box(
            low=-1.7976931348623157e308,
            high=1.7976931348623157e308,
            shape=(self.state_dim,),
        )
        self.A = 0.2 * np.array([[-1.0, -1.0, 1.0], [2.0, 0.0, 2.0], [0.0, -1.0, 2.0]])
        self.B = 0.2 * np.array([[2.0, 2.0], [2.0, 2.0], [0.0, 1.0]])
        self.Q = 0.2 * np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.5], [0.0, 0.5, 1.0]])
        self.R = 0.2 * np.array([[1.0, -1.0], [-1.0, 2.0]])
        S = np.zeros((self.state_dim, self.action_dim))
        # this matrix should be positive definite
        check_mat = np.vstack((np.hstack((self.Q, S)), np.hstack((S.T, self.R))))
        assert self.is_pos_def(check_mat)
        logger.info(
            f"Initialized Linear Dynamics Environment:\n"
            f"A:\n{self.A}\nB:\n{self.B}\nQ:\n{self.Q}\nR:\n{self.R}\n"
        )

    @staticmethod
    def is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)

    def reset(self):
        self.state = np.random.randint(low=-1, high=2, size=(self.state_dim,)).astype(
            float
        )
        self.step_cnt = 0
        return self.state

    def step(self, action):
        assert len(action) == self.action_dim
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = action.reshape((self.action_dim, 1))
        state = self.state.reshape((self.state_dim, 1))
        next_state = (self.A.dot(state) + self.B.dot(action)).squeeze()
        # add negative sign because we want to maximize the rewards,
        # while an LQR solution minimizes costs by convention
        reward = -(
            (
                state.T.dot(self.Q).dot(state) + action.T.dot(self.R).dot(action)
            ).squeeze()
        )
        self.step_cnt += 1
        terminal = False
        if self.step_cnt >= self.max_steps:
            terminal = True
        self.state = next_state
        return next_state, reward, terminal, None
