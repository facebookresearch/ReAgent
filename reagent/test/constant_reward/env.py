#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import List, Tuple

import numpy as np
import torch
from ml.rl.test.base.utils import default_normalizer
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
        self.gamma = 0.95
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
        List[np.ndarray],
    ]:

        states: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        rewards: List[int] = []
        next_states: List[np.ndarray] = []
        next_actions: List[np.ndarray] = []
        terminals: List[bool] = []
        possible_actions: List[np.ndarray] = []
        possible_next_actions: List[np.ndarray] = []

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
            possible_actions.append(np.ones(self.action_dims))
            rewards.append(self.const_reward)  # constant reward
            next_states.append(next_state)
            next_actions.append(next_action)
            terminals.append(False)
            possible_next_actions.append(np.ones(self.action_dims))

            state = next_state
            action = next_action

        return (
            states,
            actions,
            rewards,
            next_states,
            next_actions,
            terminals,
            possible_actions,
            possible_next_actions,
        )

    def preprocess_samples_discrete(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[int],
        next_states: List[np.ndarray],
        next_actions: List[np.ndarray],
        terminals: List[bool],
        possible_actions: List[np.ndarray],
        possible_next_actions: List[np.ndarray],
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
                possible_actions,
                possible_next_actions,
            )
        )
        self.np_random.shuffle(merged)
        states, actions, rewards, next_states, next_actions, terminals, possible_actions, possible_next_actions = zip(
            *merged
        )

        not_terminal = np.logical_not(terminals).reshape(-1, 1)
        time_diffs = torch.ones([len(states), 1], dtype=torch.float32)

        tdps = []
        for start in range(0, len(states), minibatch_size):
            end = start + minibatch_size
            if end > len(states):
                break
            tdps.append(
                TrainingDataPage(
                    states=torch.tensor(states[start:end], dtype=torch.float32),
                    actions=torch.tensor(actions[start:end], dtype=torch.float32),
                    propensities=torch.ones([end - start, 1], dtype=torch.float32),
                    rewards=torch.tensor(
                        rewards[start:end], dtype=torch.float32
                    ).reshape(-1, 1),
                    next_states=torch.tensor(
                        next_states[start:end], dtype=torch.float32
                    ),
                    next_actions=torch.tensor(
                        next_actions[start:end], dtype=torch.float32
                    ),
                    possible_actions_mask=torch.tensor(
                        possible_actions[start:end], dtype=torch.float32
                    ),
                    possible_next_actions_mask=torch.tensor(
                        possible_next_actions[start:end], dtype=torch.float32
                    ),
                    not_terminal=torch.tensor(
                        not_terminal[start:end].astype(np.float32), dtype=torch.float32
                    ),
                    time_diffs=time_diffs[start:end],
                )
            )
            tdps[-1].set_type(torch.FloatTensor)
        return tdps
