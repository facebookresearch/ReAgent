#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import random
from typing import Tuple, Dict, List

from caffe2.python import core, workspace

from ml.rl.preprocessing.caffe_utils import dict_list_to_blobs
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet
from ml.rl.test.utils import default_normalizer
from ml.rl.test.gridworld.gridworld_base import GridworldBase
from ml.rl.training.training_data_page import \
    TrainingDataPage


class GridworldContinuous(GridworldBase):
    @property
    def normalization_action(self):
        return default_normalizer(
            [
                x
                for x in list(
                    range(self.num_states, self.num_states + self.num_actions)
                )
            ]
        )

    def generate_samples(
        self, num_transitions, epsilon, with_possible=True
    ) -> Tuple[List[Dict[int, float]], List[Dict[int, float]], List[float],
               List[Dict[int, float]], List[Dict[int, float]], List[bool],
               List[List[Dict[int, float]]], List[Dict[int, float]]]:
        states, actions, rewards, next_states, next_actions, is_terminals,\
            possible_next_actions, reward_timelines =\
            self.generate_samples_discrete(
                num_transitions, epsilon, with_possible)
        continuous_actions = [
            {
                (self.ACTIONS.index(a) + self.num_states): 1.0
            } for a in actions
        ]
        continuous_next_actions = [
            {
                (self.ACTIONS.index(a) + self.num_states): 1.0
            } if a is not '' else {} for a in next_actions
        ]
        continuous_possible_next_actions = []
        for possible_next_action in possible_next_actions:
            continuous_possible_next_actions.append(
                [
                    {
                        (self.ACTIONS.index(a) + self.num_states): 1.0
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
        states: List[Dict[int, float]],
        actions: List[Dict[int, float]],
        rewards: List[float],
        next_states: List[Dict[int, float]],
        next_actions: List[Dict[int, float]],
        is_terminals: List[bool],
        possible_next_actions: List[List[Dict[int, float]]],
        reward_timelines: List[Dict[int, float]],
    ) -> TrainingDataPage:
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

        net = core.Net('gridworld_preprocessing')
        preprocessor = PreprocessorNet(net, True)
        lengths, keys, values = dict_list_to_blobs(states, 'states')
        state_matrix, _ = preprocessor.normalize_sparse_matrix(
            lengths,
            keys,
            values,
            self.normalization,
            'state_norm',
        )
        lengths, keys, values = dict_list_to_blobs(next_states, 'next_states')
        next_state_matrix, _ = preprocessor.normalize_sparse_matrix(
            lengths,
            keys,
            values,
            self.normalization,
            'next_state_norm',
        )
        lengths, keys, values = dict_list_to_blobs(actions, 'action')
        action_matrix, _ = preprocessor.normalize_sparse_matrix(
            lengths,
            keys,
            values,
            self.normalization_action,
            'action_norm',
        )
        lengths, keys, values = dict_list_to_blobs(next_actions, 'next_action')
        next_action_matrix, _ = preprocessor.normalize_sparse_matrix(
            lengths,
            keys,
            values,
            self.normalization_action,
            'next_action_norm',
        )
        rewards = np.array(rewards, dtype=np.float32)

        pnas_lengths = []
        pnas_flat = []
        for pnas in possible_next_actions:
            pnas_lengths.append(len(pnas))
            pnas_flat.extend(pnas)
        lengths, keys, values = dict_list_to_blobs(
            pnas_flat, 'possible_next_actions'
        )
        pnas_lengths = np.array(pnas_lengths, dtype=np.int32)
        possible_next_actions_matrix, _ = preprocessor.normalize_sparse_matrix(
            lengths,
            keys,
            values,
            self.normalization_action,
            'possible_next_action_norm',
        )
        workspace.RunNetOnce(net)

        return TrainingDataPage(
            states=workspace.FetchBlob(state_matrix),
            actions=workspace.FetchBlob(action_matrix),
            rewards=rewards,
            next_states=workspace.FetchBlob(next_state_matrix),
            next_actions=workspace.FetchBlob(next_action_matrix),
            possible_next_actions=(
                workspace.FetchBlob(possible_next_actions_matrix),
                pnas_lengths,
            ),
            reward_timelines=reward_timelines,
        )

    def true_values_for_sample(
        self, states, actions, assume_optimal_policy: bool
    ):
        string_actions = []
        for action in actions:
            string_actions.append(
                self.ACTIONS[int(list(action.keys())[0]) - self.num_states]
            )
        return GridworldBase.true_values_for_sample(
            self, states, string_actions, assume_optimal_policy
        )
