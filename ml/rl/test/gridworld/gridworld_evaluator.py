#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List, Union

import numpy as np
import numpy.testing as npt
import torch
from ml.rl.caffe_utils import softmax
from ml.rl.evaluation.evaluator import Evaluator
from ml.rl.test.gridworld.gridworld import Gridworld
from ml.rl.test.gridworld.gridworld_base import GridworldBase
from ml.rl.test.gridworld.gridworld_continuous import GridworldContinuous


logger = logging.getLogger(__name__)


class GridworldEvaluator(Evaluator):
    SOFTMAX_TEMPERATURE = 1e-6
    ABS_ERR_THRES = 0.02

    def __init__(
        self,
        env: Union[Gridworld, GridworldContinuous],
        assume_optimal_policy: bool,
        gamma: float,
    ) -> None:
        super().__init__(None, gamma, None, metrics_to_score=["reward"])

        self._env = env
        self.mc_loss: List[float] = []

        if assume_optimal_policy:
            samples = env.generate_samples(10000, 0.25, gamma)
        else:
            samples = env.generate_samples(10000, 1.0, gamma)
        self.logged_states = samples.states
        self.logged_actions = samples.actions
        self.logged_propensities = np.array(samples.action_probabilities).reshape(-1, 1)
        self.logged_terminals = np.array(samples.terminals).reshape(-1, 1)
        # Create integer logged actions
        self.logged_actions_int: List[int] = []
        for action in self.logged_actions:
            self.logged_actions_int.append(self._env.action_to_index(action))

        self.logged_actions_one_hot = np.zeros(
            [len(self.logged_actions), len(env.ACTIONS)], dtype=np.float32
        )
        for i, action in enumerate(self.logged_actions):
            self.logged_actions_one_hot[i, env.action_to_index(action)] = 1

        self.logged_values = env.true_values_for_sample(
            self.logged_states, self.logged_actions, assume_optimal_policy
        )
        self.logged_rewards = env.true_rewards_for_sample(
            self.logged_states, self.logged_actions
        )

        self.estimated_ltv_values = np.zeros(
            [len(self.logged_states), len(self._env.ACTIONS)], dtype=np.float32
        )
        for action in range(len(self._env.ACTIONS)):
            self.estimated_ltv_values[:, action] = self._env.true_values_for_sample(
                self.logged_states,
                [self._env.index_to_action(action)] * len(self.logged_states),
                True,
            ).flatten()

        self.estimated_reward_values = np.zeros(
            [len(self.logged_states), len(self._env.ACTIONS)], dtype=np.float32
        )
        for action in range(len(self._env.ACTIONS)):
            self.estimated_reward_values[:, action] = self._env.true_rewards_for_sample(
                self.logged_states,
                [self._env.index_to_action(action)] * len(self.logged_states),
            ).flatten()

    def _split_int_and_float_features(self, features):
        float_features = []
        for example in features:
            float_dict = {}
            for k, v in example.items():
                float_dict[k] = v
            float_features.append(float_dict)
        return float_features

    def evaluate(self, predictor):
        prediction_string = predictor.predict(self.logged_states)

        # Convert action string to integer
        prediction = np.zeros(
            [len(prediction_string), len(self._env.ACTIONS)], dtype=np.float32
        )
        for x in range(len(self.logged_states)):
            for action_index, action in enumerate(self._env.ACTIONS):
                prediction[x][action_index] = prediction_string[x].get(action, 1e-9)

        # Print out scores using all states
        all_states = []
        for x in self._env.STATES:
            all_states.append(self._env.state_to_features(x))
        all_states_prediction_string = predictor.predict(all_states)
        all_states_prediction = np.zeros(
            [len(all_states_prediction_string), len(self._env.ACTIONS)],
            dtype=np.float32,
        )
        for x in range(len(all_states)):
            for action_index, action in enumerate(self._env.ACTIONS):
                all_states_prediction[x][action_index] = all_states_prediction_string[
                    x
                ].get(action, 1e-9)

        self.evaluate_predictions(prediction, all_states_prediction)

    def evaluate_predictions(self, prediction, all_states_prediction):
        for i, a in enumerate(self._env.ACTIONS):
            print("Predicted Q-values for all states and action={}".format(a))
            print(all_states_prediction[:, i].reshape(5, 5), "\n")

        error_sum = 0.0
        num_error_prints = 0
        for x in range(len(self.logged_states)):
            int_action = self._env.action_to_index(self.logged_actions[x])
            logged_value = self.logged_values[x][0]
            target_value = prediction[x][int_action]
            error = abs(logged_value - target_value)
            if num_error_prints < 10 and error > self.ABS_ERR_THRES * (
                GridworldBase.REWARD_SCALE ** 2
            ):
                print(
                    "GOT {}-th STATE (POS: {}) and ACTION {} WRONG. Logged Q-Value: {}, Predicted Q-Value: {}".format(
                        x,
                        self._env._pos(list(self.logged_states[x].keys())[0]),
                        self.logged_actions[x],
                        logged_value,
                        target_value,
                    )
                )
                num_error_prints += 1
                if num_error_prints == 10:
                    print("MAX ERRORS PRINTED")
            error_sum += error
        error_mean = error_sum / float(len(self.logged_states))

        logger.info("EVAL ERROR: {0:.3f}".format(error_mean))
        self.mc_loss.append(error_mean)


class GridworldContinuousEvaluator(GridworldEvaluator):
    def evaluate(self, predictor):
        prediction_single_action = predictor.predict(
            float_state_features=self.logged_states, actions=self.logged_actions
        )

        # Convert action string to integer
        prediction = np.zeros(
            [len(prediction_single_action), len(self._env.ACTIONS)], dtype=np.float32
        )
        for x in range(len(self.logged_states)):
            for action_index, _ in enumerate(self._env.ACTIONS):
                if action_index == self._env.action_to_index(self.logged_actions[x]):
                    prediction[x][action_index] = prediction_single_action[x]["Q"]
                else:
                    prediction[x][action_index] = 1e-9

        # Print out scores using all states
        all_states = []
        all_actions = []
        for x in self._env.STATES:
            for y in self._env.ACTIONS:
                all_states.append(self._env.state_to_features(x))
                all_actions.append(self._env.action_to_features(y))

        all_states_prediction_string = predictor.predict(all_states, all_actions)
        all_states_prediction = np.zeros(
            [len(self._env.STATES), len(self._env.ACTIONS)], dtype=np.float32
        )
        c = 0
        for x in self._env.STATES:
            for y in range(len(self._env.ACTIONS)):
                all_states_prediction[x][y] = all_states_prediction_string[c]["Q"]
                c += 1
        return self.evaluate_predictions(prediction, all_states_prediction)


class GridworldDDPGEvaluator(GridworldContinuousEvaluator):
    def __init__(self, env, gamma) -> None:
        super().__init__(env, assume_optimal_policy=True, gamma=gamma)
        self.optimal_policy_samples = self._env.generate_samples(100, 0.0, self.gamma)

    def evaluate_actor(self, actor, thres: float = 0.2):
        first_states = self.logged_states[0:1000]
        actor_prediction = actor.predict(first_states)

        res_msg = (
            "num of state, num of error, state pos, pred act, "
            "optimal act, match, predicted probabilities\n"
        )
        num_of_error = 0
        for i, (state, prediction) in enumerate(zip(first_states, actor_prediction)):
            state_pos = self._env._pos(list(state.keys())[0])
            optimal_actions = self._env._optimal_actions[state_pos]
            top_prediction_index = int(
                max(prediction, key=(lambda key: prediction[key]))
            )
            top_prediction = self._env.ACTIONS[
                self._env.action_to_index({top_prediction_index: 1.0})
            ]
            if top_prediction not in optimal_actions:
                num_of_error += 1
            res_msg += "{:>12}, {:>12}, {:>9}, {:>8}, {:>11}, {:>5}, {}\n".format(
                i,
                num_of_error,
                str(state_pos),
                top_prediction,
                str(optimal_actions),
                str(top_prediction in optimal_actions),
                str(prediction),
            )

        mae = float(num_of_error) / len(first_states)
        logger.info("MAE of optimal action matching: {}".format(mae))
        logger.info("optimal action matching result:\n{}".format(res_msg))
        if mae > thres:
            logger.error(
                "High MAE of optimal action matching! MAE: {}, Threshold: {}".format(
                    mae, thres
                )
            )
        self.mc_loss.append(mae)
        return mae

    def evaluate(self, predictor):
        return self.evaluate_critic(predictor)

    def ensure_actors_match(self, actor1, actor2):
        optimal_policy_states = self.optimal_policy_samples.states
        optimal_policy_actions = self.optimal_policy_samples.actions
        optimal_policy_actions_int: List[int] = []
        for action in optimal_policy_actions:
            optimal_policy_actions_int.append(self._env.action_to_index(action))
        optimal_policy_actions_int = np.array(optimal_policy_actions_int).reshape(-1, 1)
        actor1_predictions = actor1.predict(optimal_policy_states)
        actor2_predictions = actor2.predict(optimal_policy_states)
        for pred1, pred2 in zip(actor1_predictions, actor2_predictions):
            assert set(pred1.keys()) == set(pred2.keys())
            for action_feature_id in pred1.keys():
                npt.assert_allclose(
                    pred1[action_feature_id], pred2[action_feature_id], atol=1e-2
                )

    def evaluate_critic(self, critic):
        return super().evaluate(critic)
