#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class GridworldEvaluator(object):
    def __init__(
        self,
        env,
        assume_optimal_policy: bool,
    ) -> None:
        self._test_states, self._test_actions, _, _, _, _, _, _ = env.generate_samples(
            100000, 1.0
        )
        self._test_values = env.true_values_for_sample(
            self._test_states, self._test_actions, assume_optimal_policy
        )
        self._env = env

    def evaluate(self, predictor):
        examples_by_feature = {}
        for x in range(self._env.num_states):
            examples_by_feature[str(x)] = np.zeros(
                [len(self._test_states)], dtype=np.float32
            )
        for i, test_state in enumerate(self._test_states):
            int_state = list(test_state.keys())[0]
            examples_by_feature[str(int_state)][i] = 1.0
        prediction = predictor.predict(examples_by_feature)
        error_sum = 0.0
        for x in range(len(self._test_states)):
            ground_truth = self._test_values[x]
            predicted_value = prediction[self._test_actions[x]][x][0]
            error_sum += abs(ground_truth - predicted_value)
        print('EVAL ERROR', error_sum / float(len(self._test_states)))
        return error_sum / float(len(self._test_states))


class GridworldContinuousEvaluator(GridworldEvaluator):
    def evaluate(self, predictor):
        examples_by_feature = {}
        for x in range(self._env.num_states):
            examples_by_feature[str(x)] = np.zeros(
                [len(self._test_states)], dtype=np.float32
            )
        for i, test_state in enumerate(self._test_states):
            int_state = list(test_state.keys())[0]
            examples_by_feature[str(int_state)][i] = 1.0

        examples_by_action_feature = {}
        for action_name in self._env.ACTIONS:
            examples_by_action_feature[action_name] = np.zeros(
                [len(self._test_states)], dtype=np.float32
            )
        for i, test_action in enumerate(self._test_actions):
            if len(test_action) > 0:
                action = list(test_action.keys())[0]
                examples_by_action_feature[action][i] = 1.0

        prediction = predictor.predict(
            examples_by_feature, examples_by_action_feature
        )
        error_sum = 0.0
        for x in range(len(self._test_states)):
            ground_truth = self._test_values[x]
            predicted_value = prediction['Q'][x][0]
            error_sum += abs(ground_truth - predicted_value)
        print('EVAL ERROR', error_sum / float(len(self._test_states)))
        return error_sum / float(len(self._test_states))
