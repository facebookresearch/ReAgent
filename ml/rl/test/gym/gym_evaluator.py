#!/usr/bin/env python3

import numpy as np
import logging
from typing import List, Tuple

from ml.rl.training.evaluator import Evaluator
from ml.rl.training.parametric_dqn_trainer import ParametricDQNTrainer
from ml.rl.training.continuous_action_dqn_trainer import ContinuousActionDQNTrainer
from ml.rl.test.gym.open_ai_gym_environment import EnvType

logger = logging.getLogger(__name__)


class GymEvaluator(Evaluator):
    SOFTMAX_TEMPERATURE = 0.25

    __slots__ = [
        "logged_states",
        "logged_actions",
        "logged_propensites",
        "logged_rewards",
        "logged_is_terminals"
        "_env"
    ]

    def __init__(self, env, gamma, use_int_features: bool = False) -> None:
        #TODO: incorporate int features
        super(GymEvaluator, self).__init__(None, 1, gamma)

        self._env = env
        if self._env.action_type == EnvType.CONTINUOUS_ACTION:
            raise NotImplementedError()

        (self.logged_states,
         self.logged_actions,
         self.logged_propensities,
         self.logged_rewards,
         self.logged_values,
         self.logged_is_terminals
         ) = self._generate_samples(500, 0.05)

        self.logged_states = np.array(self.logged_states).astype(np.float32)
        self.logged_propensities = np.array(self.logged_propensities).reshape(-1, 1)
        self.logged_rewards = np.array(self.logged_rewards).reshape(-1, 1)
        self.logged_values = np.array(self.logged_values).reshape(-1, 1)
        self.logged_is_terminals = np.array(self.logged_is_terminals).reshape(-1, 1)

        self.logged_actions_one_hot = np.zeros(
            shape=[len(self.logged_actions), self._env.action_dim],
            dtype=np.float32
        )
        for i, action in enumerate(self.logged_actions):
            self.logged_actions_one_hot[i, action] = 1.0

    def _generate_samples(
        self, num_episodes, epsilon
    ) -> Tuple[List[object], List[int], List[float], List[float], List[float], List[bool]]:
        """
        Generate and log samples according to a random policy.
        """

        states: List[object] = []
        actions: List[int] = []
        propensities: List[float] = []
        rewards: List[float] = []
        values: List[float] = []
        is_terminals: List[bool] = []

        for _ in range(num_episodes):
            last_end = len(states) - 1

            state = self._env.transform_state(self._env.env.reset())
            states.append(state)
            values.append(0.0)
            is_terminals.append(False)
            terminal = False

            while not terminal:
                action = self._env.action_space.sample()
                propensity = 1.0 / self._env.action_dim
                actions.append(action)
                propensities.append(propensity)

                state, reward, terminal, _ = self._env.env.step(action)
                state = self._env.transform_state(state)
                rewards.append(reward)
                is_terminals.append(terminal)

                states.append(state)
                values.append(0.0)

            # reward, action, propensity for terminal state
            rewards.append(0.0)
            values[-1] = rewards[-1]
            actions.append(self._env.action_space.sample())
            propensities.append(1.0 / self._env.action_dim)

            # calculate true values
            i = len(states) - 2
            while i > last_end:
                values[i] += rewards[i] + self.gamma * values[i + 1]
                i -= 1

        return (
            states,
            actions,
            propensities,
            rewards,
            values,
            is_terminals
        )

    def evaluate(self, predictor):
        # test only float features
        predictions = predictor.predict(self.logged_states)
        estimated_reward_values = predictor.estimate_reward(self.logged_states)
        if isinstance(predictor.trainer,
        (ParametricDQNTrainer, ContinuousActionDQNTrainer)):
            predictions = predictions.reshape(
                [-1, self._env.action_dim])
            estimated_reward_values = estimated_reward_values.reshape(
                [-1, self._env.action_dim])

        value_error_sum = 0.0
        reward_error_sum = 0.0
        for i in range(len(self.logged_states)):
            logged_action = self.logged_actions[i]
            logged_value = self.logged_values[i][0]
            target_value = predictions[i][logged_action]
            value_error_sum += abs(logged_value - target_value)
            logged_reward = self.logged_rewards[i][0]
            estimated_reward = estimated_reward_values[i][logged_action]
            reward_error_sum += abs(logged_reward - estimated_reward)
        value_error_mean = value_error_sum / np.sum(np.abs(self.logged_values))
        reward_error_mean = reward_error_sum / np.sum(np.abs(self.logged_rewards))

        logger.info("EVAL Q-Value MAE ERROR: {0:.3f}".format(value_error_mean))
        self.mc_loss.append(value_error_mean)
        logger.info("EVAL REWARD MAE ERROR: {0:.3f}".format(reward_error_mean))
        self.reward_loss.append(reward_error_mean)

        target_propensities = Evaluator.softmax(
            predictions, GymEvaluator.SOFTMAX_TEMPERATURE
        )

        reward_inverse_propensity_score, reward_direct_method, reward_doubly_robust = self.doubly_robust_one_step_policy_estimation(
            self.logged_actions_one_hot,
            self.logged_rewards,
            self.logged_propensities,
            target_propensities,
            estimated_reward_values,
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

        value_inverse_propensity_score, value_direct_method, value_doubly_robust = self.doubly_robust_one_step_policy_estimation(
            self.logged_actions_one_hot,
            self.logged_values,
            self.logged_propensities,
            target_propensities,
            predictions,
        )
        self.value_inverse_propensity_score.append(value_inverse_propensity_score)
        self.value_direct_method.append(value_direct_method)
        self.value_doubly_robust.append(value_doubly_robust)

        logger.info(
            "Value Inverse Propensity Score    : {0:.3f}".format(
                value_inverse_propensity_score
            )
        )
        logger.info(
            "Value Direct Method               : {0:.3f}".format(value_direct_method)
        )
        logger.info(
            "Value One-Step Doubly Robust P.E. : {0:.3f}".format(value_doubly_robust)
        )

        sequential_doubly_robust = self.doubly_robust_sequential_policy_estimation(
            self.logged_actions_one_hot,
            self.logged_rewards,
            self.logged_is_terminals,
            self.logged_propensities,
            target_propensities,
            predictions
        )
        self.sequential_value_doubly_robust.append(sequential_doubly_robust)

        logger.info(
            "Value Sequential Doubly Robust P.E. : {0:.3f}".format(sequential_doubly_robust)
        )

        weighted_doubly_robust = self.weighted_doubly_robust_sequential_policy_estimation(
            self.logged_actions_one_hot,
            self.logged_rewards,
            self.logged_is_terminals,
            self.logged_propensities,
            target_propensities,
            predictions
        )
        self.weighted_sequential_value_doubly_robust.append(weighted_doubly_robust)

        logger.info(
            "Value Weighted Sequential Doubly Robust P.E. : {0:.3f}".format(weighted_doubly_robust)
        )

        avg_rewards, avg_discounted_rewards = self._env.run_ep_n_times(
            100, predictor, test=True
        )

        episode_starts = np.nonzero(self.logged_is_terminals.squeeze())[0] + 1
        logged_discounted_performance = (
            self.logged_values[0][0] +
            np.sum(self.logged_values[episode_starts[:-1]])
        ) / np.sum(self.logged_is_terminals)

        true_discounted_value_PE = avg_discounted_rewards / logged_discounted_performance
        self.true_discounted_value_PE.append(true_discounted_value_PE)

        logger.info(
            "True Discounted Value P.E : {0:.3f}".format(true_discounted_value_PE)
        )

        logged_performance = np.sum(self.logged_rewards) / np.sum(self.logged_is_terminals)

        true_value_PE = avg_rewards / logged_performance
        self.true_value_PE.append(true_value_PE)

        logger.info(
            "True Value P.E : {0:.3f}".format(true_value_PE)
        )
