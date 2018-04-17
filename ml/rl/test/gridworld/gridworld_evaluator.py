#!/usr/bin/env python3



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
        prediction = predictor.predict(self._test_states)
        error_sum = 0.0
        for x in range(len(self._test_states)):
            ground_truth = self._test_values[x]
            predicted_value = prediction[x][self._test_actions[x]]
            error_sum += abs(ground_truth - predicted_value)
        print('EVAL ERROR', error_sum / float(len(self._test_states)))
        return error_sum / float(len(self._test_states))


class GridworldContinuousEvaluator(GridworldEvaluator):
    def evaluate(self, predictor):
        prediction = predictor.predict(self._test_states, self._test_actions)
        error_sum = 0.0
        for x in range(len(self._test_states)):
            ground_truth = self._test_values[x]
            predicted_value = prediction[x]['Q']
            error_sum += abs(ground_truth - predicted_value)
        print('EVAL ERROR', error_sum / float(len(self._test_states)))
        return error_sum / float(len(self._test_states))
