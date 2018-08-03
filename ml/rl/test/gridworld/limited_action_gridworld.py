#!/usr/bin/env python3


from typing import Dict, List, Optional, Tuple

import numpy as np
from ml.rl.test.gridworld.gridworld_base import G, GridworldBase, S, W
from ml.rl.training.training_data_page import TrainingDataPage


class LimitedActionGridworld(GridworldBase):
    # Left, RIGHT, UP, DOWN, CHEAT
    ACTIONS: List[str] = ["L", "R", "U", "D", "C"]

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
    STATES = list(range(width * height))
    transition_noise = 0.0

    def _cheat_step(self, state) -> int:
        for _ in range(2):
            optimal_action = self.optimal_policy(state)
            state = self._no_cheat_step(state, optimal_action)
            if self.is_terminal(state):
                break
        return state

    def generate_samples(
        self, num_transitions, epsilon, with_possible=True
    ) -> Tuple[
        List[Dict[int, float]],
        List[str],
        List[float],
        List[float],
        List[Dict[int, float]],
        List[str],
        List[bool],
        List[List[str]],
        List[Dict[int, float]],
    ]:
        return self.generate_samples_discrete(num_transitions, epsilon, with_possible)

    def preprocess_samples(
        self,
        states: List[Dict[int, float]],
        actions: List[str],
        propensities: List[float],
        rewards: List[float],
        next_states: List[Dict[int, float]],
        next_actions: List[str],
        terminals: List[bool],
        possible_next_actions: List[List[str]],
        reward_timelines: Optional[List[Dict[int, float]]],
        minibatch_size: int,
    ) -> List[TrainingDataPage]:
        return self.preprocess_samples_discrete(
            states,
            actions,
            propensities,
            rewards,
            next_states,
            next_actions,
            terminals,
            possible_next_actions,
            reward_timelines,
            minibatch_size,
        )

    def possible_next_actions(self, state, ignore_terminal=False) -> List[str]:
        if ignore_terminal is False and self.is_terminal(state):
            return []
        possible_actions = GridworldBase.possible_next_actions(
            self, state, ignore_terminal
        )
        if ignore_terminal is False:
            # Also ignore cheat actions when ignoring terminal
            possible_actions.append("C")
        return possible_actions

    def step(
        self, action: str, with_possible=True
    ) -> Tuple[int, float, bool, List[str]]:
        if action == "C":
            self._state: int = self._cheat_step(self._state)
            reward = self.reward(self._state)
            possible_next_action = self.possible_next_actions(self._state)
            return (
                self._state,
                reward,
                self.is_terminal(self._state),
                possible_next_action,
            )
        else:
            return GridworldBase.step(self, action)

    def transition_probabilities(self, state, action) -> np.ndarray:
        if action == "C":
            next_state = self._cheat_step(state)
            probabilities = np.zeros((self.width * self.height,))
            probabilities[next_state] = 1
            return probabilities
        else:
            return GridworldBase.transition_probabilities(self, state, action)
