#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import logging
from typing import Dict, List, Optional, Union

import torch
from caffe2.python import core, workspace
from ml.rl.caffe_utils import C2, StackedAssociativeArray
from ml.rl.preprocessing.normalization import sort_features_by_normalization
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.preprocessing.sparse_to_dense import Caffe2SparseToDenseProcessor
from ml.rl.test.base.utils import (
    only_continuous_action_normalizer,
    only_continuous_normalizer,
)
from ml.rl.test.environment.environment import (
    MultiStepSamples,
    Samples,
    shuffle_samples,
)
from ml.rl.test.gridworld.gridworld_base import GridworldBase
from ml.rl.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


class GridworldContinuous(GridworldBase):
    @property
    def normalization_action(self):
        return only_continuous_normalizer(
            list(range(self.num_states, self.num_states + self.num_actions)),
            min_value=0,
            max_value=1,
        )

    @property
    def normalization_continuous_action(self):
        return only_continuous_action_normalizer(
            list(range(self.num_states, self.num_states + self.num_actions)),
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

    def generate_samples(
        self,
        num_transitions,
        epsilon,
        discount_factor,
        multi_steps: Optional[int] = None,
    ) -> Union[Samples, MultiStepSamples]:
        return self.generate_random_samples(
            num_transitions,
            use_continuous_action=True,
            epsilon=epsilon,
            multi_steps=multi_steps,
        )

    def preprocess_samples(
        self,
        samples: Samples,
        minibatch_size: int,
        use_gpu: bool = False,
        one_hot_action: bool = True,
        normalize_actions: bool = True,
    ) -> List[TrainingDataPage]:
        logger.info("Shuffling...")
        samples = shuffle_samples(samples)

        logger.info("Sparse2Dense...")
        net = core.Net("gridworld_preprocessing")
        C2.set_net(net)
        sparse_to_dense_processor = Caffe2SparseToDenseProcessor()
        saa = StackedAssociativeArray.from_dict_list(samples.states, "states")
        sorted_state_features, _ = sort_features_by_normalization(self.normalization)
        state_matrix, _ = sparse_to_dense_processor(sorted_state_features, saa)
        saa = StackedAssociativeArray.from_dict_list(samples.next_states, "next_states")
        next_state_matrix, _ = sparse_to_dense_processor(sorted_state_features, saa)
        sorted_action_features, _ = sort_features_by_normalization(
            self.normalization_action
        )
        saa = StackedAssociativeArray.from_dict_list(  # type: ignore
            samples.actions, "action"
        )
        action_matrix, _ = sparse_to_dense_processor(sorted_action_features, saa)
        saa = StackedAssociativeArray.from_dict_list(  # type: ignore
            samples.next_actions, "next_action"
        )
        next_action_matrix, _ = sparse_to_dense_processor(sorted_action_features, saa)
        action_probabilities = torch.tensor(
            samples.action_probabilities, dtype=torch.float32
        ).reshape(-1, 1)
        rewards = torch.tensor(samples.rewards, dtype=torch.float32).reshape(-1, 1)

        max_action_size = 4

        pnas_mask_list: List[List[int]] = []
        pnas_flat: List[Dict[str, float]] = []
        for pnas in samples.possible_next_actions:
            pnas_mask_list.append([1] * len(pnas) + [0] * (max_action_size - len(pnas)))
            pnas_flat.extend(pnas)  # type: ignore
            for _ in range(max_action_size - len(pnas)):
                pnas_flat.append({})  # Filler
        saa = StackedAssociativeArray.from_dict_list(  # type: ignore
            pnas_flat, "possible_next_actions"
        )
        pnas_mask = torch.Tensor(pnas_mask_list)

        possible_next_actions_matrix, _ = sparse_to_dense_processor(
            sorted_action_features, saa
        )

        workspace.RunNetOnce(net)

        logger.info("Preprocessing...")
        state_preprocessor = Preprocessor(self.normalization, False)
        action_preprocessor = Preprocessor(self.normalization_action, False)

        states_ndarray = workspace.FetchBlob(state_matrix)
        states_ndarray = state_preprocessor.forward(states_ndarray)

        actions_ndarray = torch.from_numpy(workspace.FetchBlob(action_matrix))
        if normalize_actions:
            actions_ndarray = action_preprocessor.forward(actions_ndarray)

        next_states_ndarray = workspace.FetchBlob(next_state_matrix)
        next_states_ndarray = state_preprocessor.forward(next_states_ndarray)

        state_pnas_tile = next_states_ndarray.repeat(1, max_action_size).reshape(
            -1, next_states_ndarray.shape[1]
        )

        next_actions_ndarray = torch.from_numpy(workspace.FetchBlob(next_action_matrix))
        if normalize_actions:
            next_actions_ndarray = action_preprocessor.forward(next_actions_ndarray)

        logged_possible_next_actions = action_preprocessor.forward(
            workspace.FetchBlob(possible_next_actions_matrix)
        )

        assert state_pnas_tile.shape[0] == logged_possible_next_actions.shape[0], (
            "Invalid shapes: "
            + str(state_pnas_tile.shape)
            + " != "
            + str(logged_possible_next_actions.shape)
        )
        logged_possible_next_state_actions = torch.cat(
            (state_pnas_tile, logged_possible_next_actions), dim=1
        )

        logger.info("Reward Timeline to Torch...")
        time_diffs = torch.ones([len(samples.states), 1])

        tdps = []
        pnas_start = 0
        logger.info("Batching...")
        for start in range(0, states_ndarray.shape[0], minibatch_size):
            end = start + minibatch_size
            if end > states_ndarray.shape[0]:
                break
            pnas_end = pnas_start + (minibatch_size * max_action_size)
            tdp = TrainingDataPage(
                states=states_ndarray[start:end],
                actions=actions_ndarray[start:end],
                propensities=action_probabilities[start:end],
                rewards=rewards[start:end],
                next_states=next_states_ndarray[start:end],
                next_actions=next_actions_ndarray[start:end],
                not_terminal=(pnas_mask[start:end, :].sum(dim=1, keepdim=True) > 0),
                time_diffs=time_diffs[start:end],
                possible_next_actions_mask=pnas_mask[start:end, :],
                possible_next_actions_state_concat=logged_possible_next_state_actions[
                    pnas_start:pnas_end, :
                ],
            )
            pnas_start = pnas_end
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
