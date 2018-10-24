#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import logging
from typing import List

import torch
from caffe2.python import core, workspace
from ml.rl.caffe_utils import C2, StackedAssociativeArray
from ml.rl.preprocessing.normalization import sort_features_by_normalization
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.preprocessing.sparse_to_dense import sparse_to_dense
from ml.rl.test.gridworld.gridworld_base import GridworldBase, Samples
from ml.rl.test.utils import default_normalizer
from ml.rl.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


class GridworldContinuous(GridworldBase):
    @property
    def normalization_action(self):
        return default_normalizer(
            [
                x
                for x in list(
                    range(self.num_states, self.num_states + self.num_actions)
                )
            ],
            min_value=0,
            max_value=1,
        )

    @property
    def min_action_range(self):
        return torch.zeros(1, self.num_actions)

    @property
    def max_action_range(self):
        return torch.ones(1, self.num_actions)

    def action_to_index(self, action):
        return int(list(action.keys())[0]) - self.num_states

    def index_to_action(self, index):
        return {(index + self.num_states): 1.0}

    def action_to_features(self, action):
        return self.index_to_action(self.ACTIONS.index(action))

    def features_to_action(self, action):
        return self.ACTIONS[self.action_to_index(action)]

    def state_to_features(self, state):
        return {state: 1.0}

    def features_to_state(self, state):
        return list(state.keys())[0]

    def generate_samples(self, num_transitions, epsilon, discount_factor) -> Samples:
        samples = self.generate_samples_discrete(
            num_transitions, epsilon, discount_factor
        )
        continuous_actions = [self.action_to_features(a) for a in samples.actions]
        continuous_next_actions = [
            self.action_to_features(a) if a is not "" else {}
            for a in samples.next_actions
        ]
        continuous_possible_actions = []
        for possible_action in samples.possible_actions:
            continuous_possible_actions.append(
                [
                    self.action_to_features(a) if a is not None else {}
                    for a in possible_action
                ]
            )
        continuous_possible_next_actions = []
        for possible_next_action in samples.possible_next_actions:
            continuous_possible_next_actions.append(
                [
                    self.action_to_features(a) if a is not None else {}
                    for a in possible_next_action
                ]
            )

        return Samples(
            mdp_ids=samples.mdp_ids,
            sequence_numbers=samples.sequence_numbers,
            states=samples.states,
            actions=continuous_actions,
            action_probabilities=samples.action_probabilities,
            rewards=samples.rewards,
            possible_actions=continuous_possible_actions,
            next_states=samples.next_states,
            next_actions=continuous_next_actions,
            terminals=samples.terminals,
            possible_next_actions=continuous_possible_next_actions,
            episode_values=samples.episode_values,
        )

    def preprocess_samples(
        self,
        samples: Samples,
        minibatch_size: int,
        use_gpu: bool = False,
        one_hot_action: bool = True,
    ) -> List[TrainingDataPage]:
        logger.info("Shuffling...")
        samples.shuffle()

        logger.info("Sparse2Dense...")
        net = core.Net("gridworld_preprocessing")
        C2.set_net(net)
        saa = StackedAssociativeArray.from_dict_list(samples.states, "states")
        sorted_state_features, _ = sort_features_by_normalization(self.normalization)
        state_matrix, _ = sparse_to_dense(
            saa.lengths, saa.keys, saa.values, sorted_state_features
        )
        saa = StackedAssociativeArray.from_dict_list(samples.next_states, "next_states")
        next_state_matrix, _ = sparse_to_dense(
            saa.lengths, saa.keys, saa.values, sorted_state_features
        )
        sorted_action_features, _ = sort_features_by_normalization(
            self.normalization_action
        )
        saa = StackedAssociativeArray.from_dict_list(samples.actions, "action")
        action_matrix, _ = sparse_to_dense(
            saa.lengths, saa.keys, saa.values, sorted_action_features
        )
        saa = StackedAssociativeArray.from_dict_list(
            samples.next_actions, "next_action"
        )
        next_action_matrix, _ = sparse_to_dense(
            saa.lengths, saa.keys, saa.values, sorted_action_features
        )
        action_probabilities = torch.tensor(
            samples.action_probabilities, dtype=torch.float32
        ).reshape(-1, 1)
        rewards = torch.tensor(samples.rewards, dtype=torch.float32).reshape(-1, 1)

        pnas_lengths_list = []
        pnas_flat: List[List[str]] = []
        for pnas in samples.possible_next_actions:
            pnas_lengths_list.append(len(pnas))
            pnas_flat.extend(pnas)
        saa = StackedAssociativeArray.from_dict_list(pnas_flat, "possible_next_actions")

        pnas_lengths = torch.tensor(pnas_lengths_list, dtype=torch.int32)
        pna_lens_blob = "pna_lens_blob"
        workspace.FeedBlob(pna_lens_blob, pnas_lengths.numpy())

        possible_next_actions_matrix, _ = sparse_to_dense(
            saa.lengths, saa.keys, saa.values, sorted_action_features
        )

        state_pnas_tile_blob = C2.LengthsTile(next_state_matrix, pna_lens_blob)

        workspace.RunNetOnce(net)

        logger.info("Preprocessing...")
        state_preprocessor = Preprocessor(self.normalization, False)
        action_preprocessor = Preprocessor(self.normalization_action, False)

        states_ndarray = workspace.FetchBlob(state_matrix)
        states_ndarray = state_preprocessor.forward(states_ndarray)

        actions_ndarray = workspace.FetchBlob(action_matrix)
        actions_ndarray = action_preprocessor.forward(actions_ndarray)

        next_states_ndarray = workspace.FetchBlob(next_state_matrix)
        next_states_ndarray = state_preprocessor.forward(next_states_ndarray)

        next_actions_ndarray = workspace.FetchBlob(next_action_matrix)
        next_actions_ndarray = action_preprocessor.forward(next_actions_ndarray)

        logged_possible_next_actions = action_preprocessor.forward(
            workspace.FetchBlob(possible_next_actions_matrix)
        )

        state_pnas_tile = state_preprocessor.forward(
            workspace.FetchBlob(state_pnas_tile_blob)
        )
        logged_possible_next_state_actions = torch.cat(
            (state_pnas_tile, logged_possible_next_actions), dim=1
        )

        logger.info("Reward Timeline to Torch...")
        possible_next_actions_ndarray = logged_possible_next_actions
        possible_next_actions_state_concat = logged_possible_next_state_actions
        time_diffs = torch.ones([len(samples.states), 1])
        episode_values = torch.tensor(
            samples.episode_values, dtype=torch.float32
        ).reshape(-1, 1)

        tdps = []
        pnas_start = 0
        logger.info("Batching...")
        for start in range(0, states_ndarray.shape[0], minibatch_size):
            end = start + minibatch_size
            if end > states_ndarray.shape[0]:
                break
            pnas_end = pnas_start + torch.sum(pnas_lengths[start:end])
            pnas = possible_next_actions_ndarray[pnas_start:pnas_end]
            pnas_concat = possible_next_actions_state_concat[pnas_start:pnas_end]
            pnas_start = pnas_end
            tdp = TrainingDataPage(
                states=states_ndarray[start:end],
                actions=actions_ndarray[start:end],
                propensities=action_probabilities[start:end],
                rewards=rewards[start:end],
                next_states=next_states_ndarray[start:end],
                next_actions=next_actions_ndarray[start:end],
                possible_next_actions=None,
                not_terminals=(pnas_lengths[start:end] > 0).reshape(-1, 1),
                episode_values=episode_values[start:end],
                time_diffs=time_diffs[start:end],
                possible_next_actions_lengths=pnas_lengths[start:end],
                possible_next_actions_state_concat=pnas_concat,
            )
            tdp.set_type(torch.cuda.FloatTensor if use_gpu else torch.FloatTensor)
            tdps.append(tdp)
        return tdps

    def true_values_for_sample(self, states, actions, assume_optimal_policy: bool):
        string_actions = []
        for action in actions:
            string_actions.append(self.features_to_action(action))
        return GridworldBase.true_values_for_sample(
            self, states, string_actions, assume_optimal_policy
        )

    def true_rewards_for_sample(self, states, actions):
        string_actions = []
        for action in actions:
            string_actions.append(self.features_to_action(action))
        return GridworldBase.true_rewards_for_sample(self, states, string_actions)
