#!/usr/bin/env python3

import logging
from typing import List

import numpy as np
from ml.rl.training.evaluator import Evaluator


logger = logging.getLogger(__name__)


class GridworldEvaluator(Evaluator):
    SOFTMAX_TEMPERATURE = 1e-6

    def __init__(
        self, env, assume_optimal_policy: bool, gamma, use_int_features: bool, samples
    ) -> None:
        super(GridworldEvaluator, self).__init__(None, 1, gamma)

        self._env = env

        if samples is None:
            if assume_optimal_policy:
                samples = env.generate_samples(200000, 0.25)
            else:
                samples = env.generate_samples(200000, 1.0)
        self.logged_states = samples.states
        self.logged_actions = samples.actions
        self.logged_propensities = np.array(samples.propensities).reshape(-1, 1)
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

        # Print out scores using all states
        all_states = []
        for x in self._env.STATES:
            all_states.append({x: 1.0})
        if self.use_int_features:
            all_states_float, all_states_int = self._split_int_and_float_features(
                all_states
            )
            all_states_prediction_string = predictor.predict(
                all_states_float, all_states_int
            )
        else:
            all_states_prediction_string = predictor.predict(all_states)
        all_states_prediction = np.zeros(
            [len(all_states_prediction_string), len(self._env.ACTIONS)],
            dtype=np.float32,
        )
        for x in range(len(all_states)):
            for action_index, action in enumerate(self._env.ACTIONS):
                all_states_prediction[x][action_index] = all_states_prediction_string[
                    x
                ][action]
        print(all_states_prediction[:, 0].reshape(5, 5), "\n")
        print(all_states_prediction[:, 1].reshape(5, 5), "\n")
        print(all_states_prediction[:, 2].reshape(5, 5), "\n")
        print(all_states_prediction[:, 3].reshape(5, 5), "\n")

        error_sum = 0.0
        num_error_prints = 0
        for x in range(len(self.logged_states)):
            logged_value = self.logged_values[x][0]
            target_value = prediction_string[x][self.logged_actions[x]]
            error = abs(logged_value - target_value)
            if num_error_prints < 10 and error > 0.2:
                print(
                    "GOT THIS STATE WRONG: ",
                    x,
                    self._env._pos(list(self.logged_states[x].keys())[0]),
                    self.logged_actions[x],
                    logged_value,
                    target_value,
                )
                num_error_prints += 1
                if num_error_prints == 10:
                    print("MAX ERRORS PRINTED")
            error_sum += error
        error_mean = error_sum / float(len(self.logged_states))

        logger.info("EVAL ERROR: {0:.3f}".format(error_mean))
        self.mc_loss.append(error_mean)

        target_propensities = Evaluator.softmax(
            prediction, GridworldEvaluator.SOFTMAX_TEMPERATURE
        )

        reward_inverse_propensity_score, reward_direct_method, reward_doubly_robust = self.doubly_robust_one_step_policy_estimation(
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
            "Reward Inverse Propensity Score              : normalized {0:.3f} raw {1:.3f}".format(
                reward_inverse_propensity_score.normalized,
                reward_inverse_propensity_score.raw,
            )
        )
        logger.info(
            "Reward Direct Method                         : normalized {0:.3f} raw {1:.3f}".format(
                reward_direct_method.normalized, reward_direct_method.raw
            )
        )
        logger.info(
            "Reward Doubly Robust P.E.                    : normalized {0:.3f} raw {1:.3f}".format(
                reward_doubly_robust.normalized, reward_doubly_robust.raw
            )
        )

        value_inverse_propensity_score, value_direct_method, value_doubly_robust = self.doubly_robust_one_step_policy_estimation(
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
            "Value Inverse Propensity Score               : normalized {0:.3f} raw {1:.3f}".format(
                value_inverse_propensity_score.normalized,
                value_inverse_propensity_score.raw,
            )
        )
        logger.info(
            "Value Direct Method                          : normalized {0:.3f} raw {1:.3f}".format(
                value_direct_method.normalized, value_direct_method.raw
            )
        )
        logger.info(
            "Value One-Step Doubly Robust P.E.            : normalized {0:.3f} raw {1:.3f}".format(
                value_doubly_robust.normalized, value_doubly_robust.raw
            )
        )

        sequential_doubly_robust = self.doubly_robust_sequential_policy_estimation(
            self.logged_actions_one_hot,
            self.logged_rewards,
            self.logged_terminals,
            self.logged_propensities,
            target_propensities,
            self.estimated_ltv_values,
        )
        self.value_sequential_doubly_robust.append(sequential_doubly_robust)

        logger.info(
            "Value Sequential Doubly Robust P.E.          : normalized {0:.3f} raw {1:.3f}".format(
                sequential_doubly_robust.normalized, sequential_doubly_robust.raw
            )
        )

        weighted_doubly_robust = self.weighted_doubly_robust_sequential_policy_estimation(
            self.logged_actions_one_hot,
            self.logged_rewards,
            self.logged_terminals,
            self.logged_propensities,
            target_propensities,
            self.estimated_ltv_values,
            num_j_steps=1,
            whether_self_normalize_importance_weights=True,
        )
        self.value_weighted_doubly_robust.append(weighted_doubly_robust)

        logger.info(
            "Value Weighted Sequential Doubly Robust P.E. : noramlized {0:.3f} raw {1:.3f}".format(
                weighted_doubly_robust.normalized, weighted_doubly_robust.raw
            )
        )

        magic_doubly_robust = self.weighted_doubly_robust_sequential_policy_estimation(
            self.logged_actions_one_hot,
            self.logged_rewards,
            self.logged_terminals,
            self.logged_propensities,
            target_propensities,
            self.estimated_ltv_values,
            num_j_steps=GridworldEvaluator.NUM_J_STEPS_FOR_MAGIC_ESTIMATOR,
            whether_self_normalize_importance_weights=True,
        )
        self.value_magic_doubly_robust.append(magic_doubly_robust)

        logger.info(
            "Value Magic Doubly Robust P.E.               : normalized {0:.3f} raw {1:.3f}".format(
                magic_doubly_robust.normalized, magic_doubly_robust.raw
            )
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

        # Print out scores using all states
        all_states = []
        all_actions = []
        for x in self._env.STATES:
            for y in self._env.ACTIONS:
                all_states.append(self._env.state_to_features(x))
                all_actions.append(self._env.action_to_features(y))
        if self.use_int_features:
            all_states_float, all_states_int = self._split_int_and_float_features(
                all_states
            )
            all_states_prediction_string = predictor.predict(
                all_states_float, all_states_int, all_actions
            )
        else:
            all_states_prediction_string = predictor.predict(
                all_states, None, all_actions
            )
        all_states_prediction = np.zeros(
            [len(self._env.STATES), len(self._env.ACTIONS)], dtype=np.float32
        )
        c = 0
        for x in self._env.STATES:
            for y in range(len(self._env.ACTIONS)):
                all_states_prediction[x][y] = all_states_prediction_string[c]["Q"]
                c += 1
        print(all_states_prediction[:, 0].reshape(5, 5), "\n")
        print(all_states_prediction[:, 1].reshape(5, 5), "\n")
        print(all_states_prediction[:, 2].reshape(5, 5), "\n")
        print(all_states_prediction[:, 3].reshape(5, 5), "\n")

        error_sum = 0.0
        num_error_prints = 0
        for x in range(len(self.logged_states)):
            logged_value = self.logged_values[x][0]
            target_value = prediction[x]["Q"]
            error = abs(logged_value - target_value)
            error_sum += error
            if num_error_prints < 10 and error > 0.2:
                print(
                    "GOT THIS STATE WRONG: ",
                    x,
                    self._env._pos(list(self.logged_states[x].keys())[0]),
                    self.logged_actions[x],
                    logged_value,
                    target_value,
                )
                num_error_prints += 1
                if num_error_prints == 10:
                    print("MAX ERRORS PRINTED")
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
