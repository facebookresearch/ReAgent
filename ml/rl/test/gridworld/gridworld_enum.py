#!/usr/bin/env python3


from typing import Dict, List, Tuple

from ml.rl.preprocessing.normalization import NormalizationParameters

from ml.rl.test.gridworld.gridworld import Gridworld


class GridworldEnum(Gridworld):
    @property
    def num_states(self):
        return 1

    @property
    def normalization(self):
        return {
            0:
            NormalizationParameters(
                feature_type="ENUM",
                boxcox_lambda=None,
                boxcox_shift=None,
                mean=None,
                stddev=None,
                possible_values=list(range(len(self.STATES))),
                quantiles=None,
            )
        }

    def generate_samples(
        self, num_transitions, epsilon, with_possible=True
    ) -> Tuple[List[Dict[int, float]], List[str], List[float], List[
        Dict[int, float]
    ], List[str], List[bool], List[List[str]], List[Dict[int, float]]]:
        states, actions, rewards, next_states, next_actions, is_terminals, \
            possible_next_actions, reward_timelines = Gridworld.generate_samples(
                self, num_transitions, epsilon, with_possible)
        enum_states = []
        for state in states:
            enum_states.append({0: float(list(state.keys())[0])})
        enum_next_states = []
        for state in next_states:
            enum_next_states.append({0: float(list(state.keys())[0])})
        return (
            enum_states, actions, rewards, enum_next_states, next_actions,
            is_terminals, possible_next_actions, reward_timelines
        )

    def true_values_for_sample(
        self, enum_states, actions, assume_optimal_policy: bool
    ):
        states = []
        for state in enum_states:
            states.append({int(list(state.values())[0]): 1})
        return Gridworld.true_values_for_sample(
            self, states, actions, assume_optimal_policy
        )
