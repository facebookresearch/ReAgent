#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import Dict, List, Tuple

from ml.rl.preprocessing.normalization import NormalizationParameters
from ml.rl.test.gridworld.gridworld import Gridworld, Samples


class GridworldEnum(Gridworld):
    @property
    def num_states(self):
        return 1

    @property
    def normalization(self):
        return {
            0: NormalizationParameters(
                feature_type="ENUM",
                boxcox_lambda=None,
                boxcox_shift=None,
                mean=None,
                stddev=None,
                possible_values=list(range(len(self.STATES))),
                quantiles=None,
                min_value=None,
                max_value=None,
            )
        }

    def generate_samples(self, num_transitions, epsilon, discount_factor) -> Samples:
        samples = Gridworld.generate_samples(
            self, num_transitions, epsilon, discount_factor
        )
        enum_states = []
        for state in samples.states:
            enum_states.append({0: float(list(state.keys())[0])})
        enum_next_states = []
        for state in samples.next_states:
            enum_next_states.append({0: float(list(state.keys())[0])})
        return Samples(
            mdp_ids=samples.mdp_ids,
            sequence_numbers=samples.sequence_numbers,
            states=enum_states,
            actions=samples.actions,
            action_probabilities=samples.action_probabilities,
            rewards=samples.rewards,
            possible_actions=samples.possible_actions,
            next_states=enum_next_states,
            next_actions=samples.next_actions,
            terminals=samples.terminals,
            possible_next_actions=samples.possible_next_actions,
            episode_values=samples.episode_values,
        )

    def true_values_for_sample(self, enum_states, actions, assume_optimal_policy: bool):
        states = []
        for state in enum_states:
            states.append({int(list(state.values())[0]): 1})
        return Gridworld.true_values_for_sample(
            self, states, actions, assume_optimal_policy
        )

    def true_rewards_for_sample(self, enum_states, actions):
        states = []
        for state in enum_states:
            states.append({int(list(state.values())[0]): 1})
        return Gridworld.true_rewards_for_sample(self, states, actions)
