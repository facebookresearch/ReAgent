#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import random
from typing import Tuple, Dict, List

from caffe2.python import core, workspace

from ml.rl.caffe_utils import C2, StackedAssociativeArray, StackedArray
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
        minibatch_size: int,
    ) -> List[TrainingDataPage]:
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
        C2.set_net(net)
        preprocessor = PreprocessorNet(net, True)
        saa = StackedAssociativeArray.from_dict_list(states, 'states')
        state_matrix, _ = preprocessor.normalize_sparse_matrix(
            saa.lengths,
            saa.keys,
            saa.values,
            self.normalization,
            'state_norm',
        )
        saa = StackedAssociativeArray.from_dict_list(next_states, 'next_states')
        next_state_matrix, _ = preprocessor.normalize_sparse_matrix(
            saa.lengths,
            saa.keys,
            saa.values,
            self.normalization,
            'next_state_norm',
        )
        saa = StackedAssociativeArray.from_dict_list(actions, 'action')
        action_matrix, _ = preprocessor.normalize_sparse_matrix(
            saa.lengths,
            saa.keys,
            saa.values,
            self.normalization_action,
            'action_norm',
        )
        saa = StackedAssociativeArray.from_dict_list(
            next_actions, 'next_action'
        )
        next_action_matrix, _ = preprocessor.normalize_sparse_matrix(
            saa.lengths,
            saa.keys,
            saa.values,
            self.normalization_action,
            'next_action_norm',
        )
        rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)

        pnas_lengths_list = []
        pnas_flat = []
        for pnas in possible_next_actions:
            pnas_lengths_list.append(len(pnas))
            pnas_flat.extend(pnas)
        saa = StackedAssociativeArray.from_dict_list(
            pnas_flat, 'possible_next_actions'
        )
        pnas_lengths = np.array(pnas_lengths_list, dtype=np.int32)
        possible_next_actions_matrix, _ = preprocessor.normalize_sparse_matrix(
            saa.lengths,
            saa.keys,
            saa.values,
            self.normalization_action,
            'possible_next_action_norm',
        )
        workspace.RunNetOnce(net)

        states_ndarray = workspace.FetchBlob(state_matrix)
        actions_ndarray = workspace.FetchBlob(action_matrix)
        next_states_ndarray = workspace.FetchBlob(next_state_matrix)
        next_actions_ndarray = workspace.FetchBlob(next_action_matrix)
        possible_next_actions_ndarray = workspace.FetchBlob(
            possible_next_actions_matrix
        )
        tdps = []
        pnas_start = 0
        for start in range(0, states_ndarray.shape[0], minibatch_size):
            end = start + minibatch_size
            if end > states_ndarray.shape[0]:
                break
            pnas_end = pnas_start + np.sum(pnas_lengths[start:end])
            pnas = possible_next_actions_ndarray[pnas_start:pnas_end]
            pnas_start = pnas_end
            tdps.append(
                TrainingDataPage(
                    states=states_ndarray[start:end],
                    actions=actions_ndarray[start:end],
                    rewards=rewards[start:end],
                    next_states=next_states_ndarray[start:end],
                    next_actions=next_actions_ndarray[start:end],
                    possible_next_actions=StackedArray(
                        pnas_lengths[start:end], pnas
                    ),
                    not_terminals=(pnas_lengths[start:end] > 0).reshape(-1, 1),
                    reward_timelines=reward_timelines[start:end]
                    if reward_timelines else None,
                )
            )
        return tdps

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
