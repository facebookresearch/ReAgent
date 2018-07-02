#!/usr/bin/env python3

from typing import List

import numpy as np

from ml.rl.training.evaluator import Evaluator

import logging

logger = logging.getLogger(__name__)


class GridworldEvaluator(Evaluator):
    SOFTMAX_TEMPERATURE = 0.25

    def __init__(
        self, env, assume_optimal_policy: bool, use_int_features: bool = False
    ) -> None:
        Evaluator.__init__(self, None, 1)

        self._env = env

        samples = env.generate_samples(200000, 1.0)
        self.logged_states = samples.states
        self.logged_actions = samples.actions
        self.logged_propensities = np.array(samples.propensities).reshape(-1, 1)
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

        self.use_int_features = use_int_features

    def _split_int_and_float_features(self, features):
        float_features, int_features = [], []
        for example in features:
            float_dict, int_dict = {}, {}
            for k, v in example.items():
                if isinstance(v, int):
                    int_dict[k] = v
                else:
                    float_dict[k] = v
            float_features.append(float_dict)
            int_features.append(int_dict)
        return float_features, int_features

    def evaluate(self, predictor):
        # Test feeding float features & int features
        if self.use_int_features:
            float_features, int_features = self._split_int_and_float_features(
                self.logged_states
            )
            # Since all gridworld features are float types, swap these so
            # all inputs are now int_features for testing purpose
            float_features, int_features = int_features, float_features
            prediction_string = predictor.predict(float_features, int_features)
        # Test only feeding float features
        else:
            prediction_string = predictor.predict(self.logged_states)

        # Convert action string to integer
        prediction = np.zeros(
            [len(prediction_string), len(self._env.ACTIONS)], dtype=np.float32
        )
        for x in range(len(self.logged_states)):
            for action_index, action in enumerate(self._env.ACTIONS):
                prediction[x][action_index] = prediction_string[x][action]

        error_sum = 0.0
        for x in range(len(self.logged_states)):
            logged_value = self.logged_values[x][0]
            target_value = prediction_string[x][self.logged_actions[x]]
            error_sum += abs(logged_value - target_value)
        error_mean = error_sum / float(len(self.logged_states))

        logger.info("EVAL ERROR: {0:.3f}".format(error_mean))
        self.mc_loss.append(error_mean)

        target_propensities = Evaluator.softmax(
            prediction, GridworldEvaluator.SOFTMAX_TEMPERATURE
        )

        value_inverse_propensity_score, value_direct_method, value_doubly_robust = self.doubly_robust_policy_estimation(
            self.logged_actions_one_hot,
            self.logged_values,
            self.logged_propensities,
            target_propensities,
            self.estimated_ltv_values,
        )
        self.value_inverse_propensity_score.append(value_inverse_propensity_score)
        self.value_direct_method.append(value_direct_method)
        self.value_doubly_robust.append(value_doubly_robust)

        logger.info(
            "Value Inverse Propensity Score : {0:.3f}".format(
                value_inverse_propensity_score
            )
        )
        logger.info(
            "Value Direct Method            : {0:.3f}".format(value_direct_method)
        )
        logger.info(
            "Value Doubly Robust P.E.       : {0:.3f}".format(value_doubly_robust)
        )

        reward_inverse_propensity_score, reward_direct_method, reward_doubly_robust = self.doubly_robust_policy_estimation(
            self.logged_actions_one_hot,
            self.logged_rewards,
            self.logged_propensities,
            target_propensities,
            self.estimated_reward_values,
        )
        self.reward_inverse_propensity_score.append(reward_inverse_propensity_score)
        self.reward_direct_method.append(reward_direct_method)
        self.reward_doubly_robust.append(reward_doubly_robust)

        logger.info(
            "Reward Inverse Propensity Score: {0:.3f}".format(
                reward_inverse_propensity_score
            )
        )
        logger.info(
            "Reward Direct Method           : {0:.3f}".format(reward_direct_method)
        )
        logger.info(
            "Reward Doubly Robust P.E.      : {0:.3f}".format(reward_doubly_robust)
        )


class GridworldContinuousEvaluator(GridworldEvaluator):
    def evaluate(self, predictor):
        # Test feeding float features & int features
        if self.use_int_features:
            float_features, int_features = self._split_int_and_float_features(
                self.logged_states
            )
            # Since all gridworld features are float types, swap these so
            # all inputs are now int_features for testing purpose
            float_features, int_features = int_features, float_features
            prediction = predictor.predict(
                float_state_features=float_features,
                int_state_features=int_features,
                actions=self.logged_actions,
            )
        # Test only feeding float features
        else:
            prediction = predictor.predict(
                float_state_features=self.logged_states,
                int_state_features=None,
                actions=self.logged_actions,
            )
        error_sum = 0.0
        for x in range(len(self.logged_states)):
            ground_truth = self.logged_values[x][0]
            target_value = prediction[x]["Q"]
            error_sum += abs(ground_truth - target_value)
        logger.info(
            "EVAL ERROR {0:.3f}".format(error_sum / float(len(self.logged_states)))
        )
        return error_sum / float(len(self.logged_states))


class GridworldDDPGEvaluator(GridworldEvaluator):
    def evaluate_actor(self, actor):
        actor_prediction = actor.actor_prediction(self.logged_states)
        logger.info(
            "Actor predictions executed successfully. Sample: {}".format(
                actor_prediction
            )
        )

    def evaluate_critic(self, critic):
        critic_prediction = critic.critic_prediction(
            float_state_features=self.logged_states,
            int_state_features=None,
            actions=self.logged_actions,
        )
        error_sum = 0.0
        for x in range(len(self.logged_states)):
            ground_truth = self.logged_values[x][0]
            target_value = critic_prediction[x]
            error_sum += abs(ground_truth - target_value)
        logger.info(
            "EVAL ERROR: {0:.3f}".format(error_sum / float(len(self.logged_states)))
        )
        return error_sum / float(len(self.logged_states))
