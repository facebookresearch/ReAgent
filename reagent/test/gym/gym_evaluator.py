#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from reagent.evaluation.evaluator import Evaluator
from reagent.test.gym.open_ai_gym_environment import EnvType
from reagent.training.on_policy_predictor import OnPolicyPredictor
from reagent.training.parametric_dqn_trainer import ParametricDQNTrainer


logger = logging.getLogger(__name__)


class GymEvaluator(Evaluator):
    SOFTMAX_TEMPERATURE = 0.25

    __slots__ = [
        "logged_states",
        "logged_actions",
        "logged_propensites",
        "logged_rewards",
        "logged_terminals" "_env",
    ]

    def __init__(self, env, gamma) -> None:
        # TODO: incorporate int features
        super().__init__(None, gamma, None, 1.0)

        self._env = env
        self.mc_loss: List[float] = []
        if self._env.action_type == EnvType.CONTINUOUS_ACTION:
            return

        (
            logged_mdp_ids,
            logged_sequence_numbers,
            logged_states,
            self.logged_actions,
            logged_propensities,
            logged_rewards,
            logged_values,
            logged_terminals,
        ) = self._generate_samples(500, 0.05)

        self.logged_mdp_ids = logged_mdp_ids
        self.logged_sequence_numbers = torch.tensor(logged_sequence_numbers).reshape(
            -1, 1
        )
        self.logged_states = torch.stack(logged_states)
        self.logged_propensities = (
            torch.tensor(logged_propensities).float().reshape(-1, 1)
        )
        self.logged_rewards = torch.tensor(logged_rewards).float().reshape(-1, 1)
        self.logged_values = torch.tensor(logged_values).float().reshape(-1, 1)
        self.logged_terminals = torch.tensor(logged_terminals).reshape(-1, 1)
        self.logged_actions_one_hot = torch.zeros(
            [len(self.logged_actions), self._env.action_dim]
        )
        for i, action in enumerate(self.logged_actions):
            self.logged_actions_one_hot[i, action] = 1.0

        self.reward_loss: List[float] = []

    def _generate_samples(
        self, num_episodes, epsilon
    ) -> Tuple[
        List[str],
        List[int],
        List[torch.Tensor],
        List[int],
        List[float],
        List[float],
        List[float],
        List[bool],
    ]:
        """
        Generate and log samples according to a random policy.
        """

        mdp_ids: List[str] = []
        sequence_numbers: List[int] = []
        states: List[torch.Tensor] = []
        actions: List[int] = []
        propensities: List[float] = []
        rewards: List[float] = []
        values: List[float] = []
        terminals: List[bool] = []

        for episode in range(num_episodes):
            sequence_number = 0
            last_end = len(states) - 1

            state = self._env.transform_state(self._env.reset())
            mdp_ids.append(str(episode))
            sequence_numbers.append(sequence_number)
            states.append(state)
            values.append(0.0)
            terminals.append(False)
            terminal = False

            while not terminal:
                action = self._env.action_space.sample()
                propensity = 1.0 / self._env.action_dim
                actions.append(action)
                propensities.append(propensity)

                state, reward, terminal, _ = self._env.step(action)
                state = self._env.transform_state(state)
                rewards.append(reward)
                terminals.append(terminal)

                states.append(state)
                mdp_ids.append(str(episode))
                sequence_number += 1
                sequence_numbers.append(sequence_number)
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
            mdp_ids,
            sequence_numbers,
            states,
            actions,
            propensities,
            rewards,
            values,
            terminals,
        )

    def evaluate_on_episodes(
        self, num_episodes, predictor: Optional[OnPolicyPredictor]
    ):
        """
        Simulate real episodes and evaluate average rewards and
        discounted rewards

        :param num_episodes: number of episodes to simulate
        :param predictor: the policy to be used to run episodes. If it is None,
            use a random policy
        """
        return self._env.run_ep_n_times(num_episodes, predictor, test=True)

    def evaluate_reward_value(self, predictor):
        if self._env.action_type == EnvType.CONTINUOUS_ACTION:
            raise NotImplementedError()
        # test only float features
        if predictor.discrete_action():
            predictions = predictor.predict(self.logged_states)
            estimated_reward_values = predictor.estimate_reward(self.logged_states)
        else:
            num_states = self.logged_states.size()[0]
            states_tiled = torch.repeat_interleave(
                self.logged_states, repeats=self._env.action_dim, axis=0
            )
            action_tiled = torch.repeat_interleave(
                torch.eye(self._env.action_dim), repeats=num_states, axis=0
            )

            predictions = predictor.predict(states_tiled, action_tiled).reshape(
                [-1, self._env.action_dim]
            )
            estimated_reward_values = predictor.estimate_reward(
                states_tiled, action_tiled
            ).reshape([-1, self._env.action_dim])

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
        value_error_mean = value_error_sum / torch.sum(torch.abs(self.logged_values))
        reward_error_mean = reward_error_sum / torch.sum(torch.abs(self.logged_rewards))

        logger.info("EVAL Q-Value MAE ERROR: {0:.3f}".format(value_error_mean))
        self.mc_loss.append(value_error_mean)
        logger.info("EVAL REWARD MAE ERROR: {0:.3f}".format(reward_error_mean))
        self.reward_loss.append(reward_error_mean)

        return reward_error_mean, value_error_mean
