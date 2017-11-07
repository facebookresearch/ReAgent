#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from typing import Tuple, List

from ml.rl.test.gridworld.gridworld_base import GridworldBase, W, S, G


class LimitedActionGridworld(GridworldBase):
    # Left, RIGHT, UP, DOWN, CHEAT
    ACTIONS = ['L', 'R', 'U', 'D', 'C']

    grid = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],  #
            [0, 0, 0, G, W, 0, 0],  #
            [0, 0, W, W, W, 0, 0],  #
            [0, W, 0, S, 0, W, 0],  #
            [0, W, 0, 0, 0, W, 0],  #
            [0, 0, W, 0, W, 0, 0],  #
            [0, 0, 0, 0, 0, 0, 0],  #
        ]
    )

    width = 7
    height = 7
    STATES = [str(i) for i in range(width * height)]
    transition_noise = 0.0

    def _cheat_step(self, state):
        for _ in range(2):
            optimal_action = self.optimal_policy(state)
            state = self._no_cheat_step(state, optimal_action)
            if self.is_terminal(state):
                break
        return state

    def possible_next_actions(self, state, ignore_terminal=False):
        if ignore_terminal is False and self.is_terminal(state):
            return []
        possible_actions = GridworldBase.possible_next_actions(
            self, state, ignore_terminal
        )
        if ignore_terminal is False:
            # Also ignore cheat actions when ignoring terminal
            possible_actions.append('C')
        return possible_actions

    def step(self, action: str,
             with_possible=True) -> Tuple[int, float, bool, List[str]]:
        if action == 'C':
            self._state: int = self._cheat_step(self._state)
            reward = self.reward(self._state)
            possible_next_action = self.possible_next_actions(self._state)
            return self._state, reward, self.is_terminal(
                self._state
            ), possible_next_action
        else:
            return super(self.__class__, self).step(action)

    def transition_probabilities(self, state, action):
        if action == 'C':
            next_state = self._cheat_step(state)
            probabilities = np.zeros((self.width * self.height, ))
            probabilities[next_state] = 1
            return probabilities
        else:
            return super(self.__class__,
                         self).transition_probabilities(state, action)
