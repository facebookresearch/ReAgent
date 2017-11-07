#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import numpy as np
import random
from typing import Tuple, Dict, List

from ml.rl.test.gridworld.gridworld_base import default_normalizer
from ml.rl.test.gridworld.gridworld import Gridworld
from ml.rl.training.training_data_page import \
    TrainingDataPage


class GridworldContinuous(Gridworld):
    @property
    def normalization_action(self):
        return default_normalizer(self.ACTIONS)

    def generate_samples(
        self, num_transitions, epsilon, with_possible=True
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], List[float],
               List[Dict[str, float]], List[Dict[str, float]], List[bool],
               List[List[Dict[str, float]]], List[Dict[int, float]]]:
        states, actions, rewards, next_states, next_actions, is_terminals,\
            possible_next_actions, reward_timelines = \
            Gridworld.generate_samples(
                self, num_transitions, epsilon, with_possible)
        continuous_actions = [{a: 1.0} for a in actions]
        continuous_next_actions = [
            {
                a: 1.0
            } if a is not '' else {} for a in next_actions
        ]
        continuous_possible_next_actions = []
        for possible_next_action in possible_next_actions:
            continuous_possible_next_actions.append(
                [
                    {
                        a: 1.0
                    } if a is not None else {} for a in possible_next_action
                ]
            )

        return (
            states, continuous_actions, rewards, next_states,
            continuous_next_actions, is_terminals,
            continuous_possible_next_actions, reward_timelines
        )

    def preprocess_samples(
        self,
        states: List[Dict[str, float]],
        actions: List[Dict[str, float]],
        rewards: List[float],
        next_states: List[Dict[str, float]],
        next_actions: List[Dict[str, float]],
        is_terminals: List[bool],
        possible_next_actions: List[List[Dict[str, float]]],
        reward_timelines: List[Dict[int, float]],
    ):
        # Shuffle
        merged = list(
            zip(
                states, actions, rewards, next_states, next_actions,
                is_terminals, possible_next_actions, reward_timelines
            )
        )
        random.shuffle(merged)
        states, actions, rewards, next_states, next_actions, is_terminals, \
            possible_next_actions, reward_timelines = zip(*merged)

        x = []
        for state in states:
            a = [0.0] * self.num_states
            a[int(list(state.keys())[0])] = 1.0
            x.append(a)
        states = np.array(x, dtype=np.float32)
        x = []
        for state in next_states:
            a = [0.0] * self.num_states
            a[int(list(state.keys())[0])] = 1.0
            x.append(a)
        next_states = np.array(x, dtype=np.float32)
        x = []
        for action in actions:
            a = [0.0] * self.num_actions
            if len(action) > 0:
                a[self.ACTIONS.index(list(action.keys())[0])] = 1.0
            x.append(a)
        actions = np.array(x, dtype=np.float32)
        x = []
        for action in next_actions:
            a = [0.0] * self.num_actions
            if len(action) > 0:
                a[self.ACTIONS.index(list(action.keys())[0])] = 1.0
            x.append(a)
        next_actions = np.array(x, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)

        continuous_possible_next_actions = []
        for pnas in possible_next_actions:
            pna = []
            for action in pnas:
                a = [0.0] * self.num_actions
                if len(action) > 0:
                    a[self.ACTIONS.index(list(action.keys())[0])] = 1.0
                pna.append(a)
            continuous_possible_next_actions.append(
                np.array(pna, dtype=np.float32)
            )
        continuous_possible_next_actions = np.array(
            continuous_possible_next_actions, dtype=np.object
        )

        return TrainingDataPage(
            state_features=states,
            action=actions,
            reward=rewards,
            next_state_features=next_states,
            next_action=next_actions,
            possible_next_actions=continuous_possible_next_actions,
            reward_timelines=reward_timelines,
            ds=[datetime.date.today().strftime('%Y-%m-%d')] * len(rewards)
        )

    def true_values_for_sample(
        self, states, actions, assume_optimal_policy: bool
    ):
        string_actions = []
        for action in actions:
            string_actions.append(list(action.keys())[0])
        return Gridworld.true_values_for_sample(
            self, states, string_actions, assume_optimal_policy
        )
