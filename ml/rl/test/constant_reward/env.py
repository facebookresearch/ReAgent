#!/usr/bin/env python3

from typing import List, Tuple

import numpy as np
from ml.rl.test.utils import default_normalizer
from ml.rl.training.training_data_page import TrainingDataPage


class Env(object):
    """
    A simple environment that returns constant reward
    """

    def __init__(self, state_dims, action_dims):
        self.np_random = np.random.RandomState()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.actions = [str(i) for i in range(self.action_dims)]
        self.const_reward = 1
        self.gamma = 0.99
        assert 0 < self.gamma < 1

    def seed(self, seed):
        self.np_random.seed(seed)

    @property
    def normalization(self):
        return default_normalizer(list(range(self.state_dims)))

    def generate_samples_discrete(
        self, num_transitions, random_each_step=True, terminal=False
    ) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[int],
        List[np.ndarray],
        List[np.ndarray],
        List[bool],
        List[np.ndarray],
        List[float],
    ]:

        states: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        rewards: List[int] = []
        next_states: List[np.ndarray] = []
        next_actions: List[np.ndarray] = []
        terminals: List[bool] = []
        possible_next_actions: List[np.ndarray] = []
        episode_values: List[float] = []

        state = self.np_random.uniform(low=-1, high=-1, size=(self.state_dims))
        action = np.zeros(self.action_dims)
        action[self.np_random.randint(self.action_dims)] = 1

        for _ in range(num_transitions):
            if random_each_step:
                next_state = self.np_random.uniform(
                    low=-1, high=-1, size=(self.state_dims)
                )
                next_action = np.zeros(self.action_dims)
                next_action[self.np_random.randint(self.action_dims)] = 1
            else:
                next_state = np.array(state, copy=True)
                next_action = np.array(action, copy=True)

            states.append(state)
            actions.append(action)
            rewards.append(self.const_reward)  # constant reward
            next_states.append(next_state)
            next_actions.append(next_action)
            terminals.append(False)
            possible_next_actions.append(np.ones(self.action_dims))
            episode_values.append(1 / (1 - self.gamma))

            state = next_state
            action = next_action

        return (
            states,
            actions,
            rewards,
            next_states,
            next_actions,
            terminals,
            possible_next_actions,
            episode_values,
        )

    def preprocess_samples_discrete(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[int],
        next_states: List[np.ndarray],
        next_actions: List[np.ndarray],
        terminals: List[bool],
        possible_next_actions: List[np.ndarray],
        episode_values: List[float],
        minibatch_size: int,
    ) -> List[TrainingDataPage]:
        # Shuffle
        merged = list(
            zip(
                states,
                actions,
                rewards,
                next_states,
                next_actions,
                terminals,
                possible_next_actions,
                episode_values,
            )
        )
        self.np_random.shuffle(merged)
        states, actions, rewards, next_states, next_actions, terminals, possible_next_actions, episode_values = zip(
            *merged
        )

        not_terminals = np.logical_not(terminals).reshape(-1, 1)

        tdps = []
        for start in range(0, len(states), minibatch_size):
            end = start + minibatch_size
            if end > len(states):
                break
            tdps.append(
                TrainingDataPage(
                    states=np.array(states[start:end], dtype=np.float32),
                    actions=np.array(actions[start:end], dtype=np.float32),
                    propensities=np.ones([end - start, 1]),
                    rewards=np.array(rewards[start:end], dtype=np.float32).reshape(
                        -1, 1
                    ),
                    next_states=np.array(next_states[start:end], dtype=np.float32),
                    next_actions=np.array(next_actions[start:end], dtype=np.float32),
                    possible_next_actions=np.array(
                        possible_next_actions[start:end], dtype=np.float32
                    ),
                    episode_values=np.array(
                        episode_values[start:end], dtype=np.float32
                    ).reshape(-1, 1),
                    not_terminals=not_terminals[start:end],
                )
            )
        return tdps
