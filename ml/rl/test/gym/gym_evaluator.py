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
    DISCOUNT = 0.99

    __slots__ = [
        "logged_states",
        "logged_actions",
        "logged_propensites",
        "logged_rewards",
        "_env"
    ]

    def __init__(self, env, use_int_features: bool = False) -> None:
        #TODO: incorporate int features
        super(GymEvaluator, self).__init__(None, 1)

        self._env = env
        if self._env.action_type == EnvType.CONTINUOUS_ACTION:
            raise NotImplementedError()

        (self.logged_states,
         self.logged_actions,
         self.logged_propensities,
         self.logged_rewards,
         self.logged_values
         ) = GymEvaluator.generate_samples(self._env, 50, 0.05)

        self.logged_states = np.array(self.logged_states).astype(np.float32)
        self.logged_propensities = np.array(self.logged_propensities).reshape(-1, 1)
        self.logged_rewards = np.array(self.logged_rewards).reshape(-1, 1)
        self.logged_values = np.array(self.logged_values).reshape(-1, 1)

        self.logged_actions_one_hot = np.zeros(
            shape=[len(self.logged_actions), self._env.action_dim],
            dtype=np.float32
        )
        for i, action in enumerate(self.logged_actions):
            self.logged_actions_one_hot[i, action] = 1.0

    @staticmethod
    def generate_samples(
        env, num_episodes, epsilon
    ) -> Tuple[List[object], List[int], List[float], List[float], List[float]]:
        """
        Generate and log samples according to a random policy.
        """

        states: List[object] = []
        actions: List[int] = []
        propensities: List[float] = []
        rewards: List[float] = []
        values: List[float] = []

        for _ in range(num_episodes):
            last_end = len(states) - 1

            terminal = False
            state = env.transform_state(env.env.reset())
            states.append(state)
            values.append(0.0)

            while not terminal:
                action = env.action_space.sample()
                propensity = 1.0 / env.action_dim
                actions.append(action)
                propensities.append(propensity)

                state, reward, terminal, _ = env.env.step(action)
                state = env.transform_state(state)
                rewards.append(reward)

                states.append(state)
                values.append(0.0)

            # reward, action, propensity for terminal state
            rewards.append(0.0)
            values[-1] = rewards[-1]
            actions.append(env.action_space.sample())
            propensities.append(1.0 / env.action_dim)

            # calculate true values
            i = len(states) - 2
            while i > last_end:
                values[i] += rewards[i] + GymEvaluator.DISCOUNT * values[i + 1]
                i -= 1

        return (
            states,
            actions,
            propensities,
            rewards,
            values
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
        value_error_mean = value_error_sum / float(len(self.logged_states))
        reward_error_mean = reward_error_sum / float(len(self.logged_states))

        logger.info("EVAL Q-Value MAE ERROR: {0:.3f}".format(value_error_mean))
        self.mc_loss.append(value_error_mean)
        logger.info("EVAL REWARD MAE ERROR: {0:.3f}".format(reward_error_mean))
        self.reward_loss.append(reward_error_mean)

        target_propensities = Evaluator.softmax(
            predictions, GymEvaluator.SOFTMAX_TEMPERATURE
        )

        value_inverse_propensity_score, value_direct_method, value_doubly_robust = self.doubly_robust_policy_estimation(
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
