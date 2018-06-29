#!/usr/bin/env python3


import numpy as np
from typing import Tuple, Dict, List

from caffe2.python import core, workspace

from ml.rl.caffe_utils import C2, StackedAssociativeArray, StackedArray
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet
from ml.rl.test.utils import default_normalizer
from ml.rl.test.gridworld.gridworld_base import GridworldBase, Samples
from ml.rl.training.training_data_page import TrainingDataPage


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

    def action_to_index(self, action):
        return int(list(action.keys())[0]) - self.num_states

    def index_to_action(self, index):
        return {(index + self.num_states): 1.0}

    def action_to_features(self, action):
        return self.index_to_action(self.ACTIONS.index(action))

    def features_to_action(self, action):
        return self.ACTIONS[self.action_to_index(action)]

    def generate_samples(self, num_transitions, epsilon, with_possible=True) -> Samples:
        samples = self.generate_samples_discrete(
            num_transitions, epsilon, with_possible
        )
        continuous_actions = [self.action_to_features(a) for a in samples.actions]
        continuous_next_actions = [
            self.action_to_features(a) if a is not "" else {}
            for a in samples.next_actions
        ]
        continuous_possible_next_actions = []
        for possible_next_action in samples.possible_next_actions:
            continuous_possible_next_actions.append(
                [
                    self.action_to_features(a) if a is not None else {}
                    for a in possible_next_action
                ]
            )

        return Samples(
            states=samples.states,
            actions=continuous_actions,
            propensities=samples.propensities,
            rewards=samples.rewards,
            next_states=samples.next_states,
            next_actions=continuous_next_actions,
            is_terminal=samples.is_terminal,
            possible_next_actions=continuous_possible_next_actions,
            reward_timelines=samples.reward_timelines,
        )

    def preprocess_samples(
        self, samples: Samples, minibatch_size: int
    ) -> List[TrainingDataPage]:
        samples.shuffle()

        net = core.Net("gridworld_preprocessing")
        C2.set_net(net)
        preprocessor = PreprocessorNet(True)
        saa = StackedAssociativeArray.from_dict_list(samples.states, "states")
        state_matrix, _ = preprocessor.normalize_sparse_matrix(
            saa.lengths,
            saa.keys,
            saa.values,
            self.normalization,
            "state_norm",
            False,
            False,
        )
        saa = StackedAssociativeArray.from_dict_list(samples.next_states, "next_states")
        next_state_matrix, _ = preprocessor.normalize_sparse_matrix(
            saa.lengths,
            saa.keys,
            saa.values,
            self.normalization,
            "next_state_norm",
            False,
            False,
        )
        saa = StackedAssociativeArray.from_dict_list(samples.actions, "action")
        action_matrix, _ = preprocessor.normalize_sparse_matrix(
            saa.lengths,
            saa.keys,
            saa.values,
            self.normalization_action,
            "action_norm",
            False,
            False,
        )
        saa = StackedAssociativeArray.from_dict_list(
            samples.next_actions, "next_action"
        )
        next_action_matrix, _ = preprocessor.normalize_sparse_matrix(
            saa.lengths,
            saa.keys,
            saa.values,
            self.normalization_action,
            "next_action_norm",
            False,
            False,
        )
        propensities = np.array(samples.propensities, dtype=np.float32).reshape(-1, 1)
        rewards = np.array(samples.rewards, dtype=np.float32).reshape(-1, 1)

        pnas_lengths_list = []
        pnas_flat: List[List[str]] = []
        for pnas in samples.possible_next_actions:
            pnas_lengths_list.append(len(pnas))
            pnas_flat.extend(pnas)
        saa = StackedAssociativeArray.from_dict_list(pnas_flat, "possible_next_actions")
        pnas_lengths = np.array(pnas_lengths_list, dtype=np.int32)
        possible_next_actions_matrix, _ = preprocessor.normalize_sparse_matrix(
            saa.lengths,
            saa.keys,
            saa.values,
            self.normalization_action,
            "possible_next_action_norm",
            False,
            False,
        )
        workspace.RunNetOnce(net)

        states_ndarray = workspace.FetchBlob(state_matrix)
        actions_ndarray = workspace.FetchBlob(action_matrix)
        next_states_ndarray = workspace.FetchBlob(next_state_matrix)
        next_actions_ndarray = workspace.FetchBlob(next_action_matrix)
        possible_next_actions_ndarray = workspace.FetchBlob(
            possible_next_actions_matrix
        )
        time_diffs = np.ones(len(states_ndarray))
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
                    propensities=propensities[start:end],
                    rewards=rewards[start:end],
                    next_states=next_states_ndarray[start:end],
                    next_actions=next_actions_ndarray[start:end],
                    possible_next_actions=StackedArray(pnas_lengths[start:end], pnas),
                    not_terminals=(pnas_lengths[start:end] > 0).reshape(-1, 1),
                    reward_timelines=samples.reward_timelines[start:end]
                    if samples.reward_timelines
                    else None,
                    time_diffs=time_diffs[start:end],
                )
            )
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
