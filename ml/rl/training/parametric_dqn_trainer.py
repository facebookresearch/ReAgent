#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Tuple

import ml.rl.types as rlt
import numpy as np
import torch
import torch.nn.functional as F
from ml.rl.thrift.core.ttypes import ContinuousActionModelParameters
from ml.rl.training.dqn_trainer_base import DQNTrainerBase
from ml.rl.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


class ParametricDQNTrainer(DQNTrainerBase):
    def __init__(
        self,
        q_network,
        q_network_target,
        reward_network,
        parameters: ContinuousActionModelParameters,
        use_gpu: bool = False,
    ) -> None:
        self.double_q_learning = parameters.rainbow.double_q_learning
        self.minibatch_size = parameters.training.minibatch_size
        self.minibatches_per_step = parameters.training.minibatches_per_step or 1

        DQNTrainerBase.__init__(self, parameters, use_gpu=use_gpu)

        self.q_network = q_network
        self.q_network_target = q_network_target
        self._set_optimizer(parameters.training.optimizer)
        self.q_network_optimizer = self.optimizer_func(
            self.q_network.parameters(),
            lr=parameters.training.learning_rate,
            weight_decay=parameters.training.l2_decay,
        )

        self.reward_network = reward_network
        self.reward_network_optimizer = self.optimizer_func(
            self.reward_network.parameters(), lr=parameters.training.learning_rate
        )

    def get_detached_q_values(
        self, state, action
    ) -> Tuple[rlt.SingleQValue, rlt.SingleQValue]:
        """ Gets the q values from the model and target networks """
        with torch.no_grad():
            input = rlt.StateAction(state=state, action=action)
            q_values = self.q_network(input)
            q_values_target = self.q_network_target(input)
        return q_values, q_values_target

    def train(self, training_batch) -> None:
        if isinstance(training_batch, TrainingDataPage):
            if self.maxq_learning:
                training_batch = training_batch.as_parametric_maxq_training_batch()
            else:
                training_batch = training_batch.as_parametric_sarsa_training_batch()

        learning_input = training_batch.training_input
        self.minibatch += 1

        reward = learning_input.reward
        not_done_mask = learning_input.not_terminal

        discount_tensor = torch.full_like(reward, self.gamma)
        if self.use_seq_num_diff_as_time_diff:
            assert self.multi_steps is None
            discount_tensor = torch.pow(self.gamma, learning_input.time_diff.float())
        if self.multi_steps is not None:
            discount_tensor = torch.pow(self.gamma, learning_input.step.float())

        if self.maxq_learning:
            all_next_q_values, all_next_q_values_target = self.get_detached_q_values(
                learning_input.tiled_next_state, learning_input.possible_next_actions
            )
            # Compute max a' Q(s', a') over all possible actions using target network
            next_q_values, _ = self.get_max_q_values_with_target(
                all_next_q_values.q_value,
                all_next_q_values_target.q_value,
                learning_input.possible_next_actions_mask.float(),
            )
        else:
            # SARSA (Use the target network)
            _, next_q_values = self.get_detached_q_values(
                learning_input.next_state, learning_input.next_action
            )
            next_q_values = next_q_values.q_value

        filtered_max_q_vals = next_q_values * not_done_mask.float()

        target_q_values = reward + (discount_tensor * filtered_max_q_vals)

        # Get Q-value of action taken
        current_state_action = rlt.StateAction(
            state=learning_input.state, action=learning_input.action
        )
        q_values = self.q_network(current_state_action).q_value
        self.all_action_scores = q_values.detach()

        value_loss = self.q_network_loss(q_values, target_q_values)
        self.loss = value_loss.detach()

        value_loss.backward()
        self._maybe_run_optimizer(self.q_network_optimizer, self.minibatches_per_step)

        # Use the soft update rule to update target network
        self._maybe_soft_update(
            self.q_network, self.q_network_target, self.tau, self.minibatches_per_step
        )

        # get reward estimates
        reward_estimates = self.reward_network(current_state_action).q_value
        reward_loss = F.mse_loss(reward_estimates, reward)
        reward_loss.backward()
        self._maybe_run_optimizer(
            self.reward_network_optimizer, self.minibatches_per_step
        )

        self.loss_reporter.report(
            td_loss=self.loss,
            reward_loss=reward_loss,
            model_values_on_logged_actions=self.all_action_scores,
        )

    def internal_prediction(self, state, action):
        """
        Only used by Gym
        """
        self.q_network.eval()
        with torch.no_grad():
            state = torch.from_numpy(np.array(state)).type(self.dtype)
            action = torch.from_numpy(np.array(action)).type(self.dtype)
            q_values = self.q_network(
                rlt.StateAction(
                    state=rlt.FeatureVector(float_features=state),
                    action=rlt.FeatureVector(float_features=action),
                )
            )
        self.q_network.train()
        return q_values.q_value.cpu().data.numpy()

    def internal_reward_estimation(self, state, action):
        """
        Only used by Gym
        """
        self.reward_network.eval()
        with torch.no_grad():
            state = torch.from_numpy(np.array(state)).type(self.dtype)
            action = torch.from_numpy(np.array(action)).type(self.dtype)
            reward_estimates = self.reward_network(
                rlt.StateAction(
                    state=rlt.FeatureVector(float_features=state),
                    action=rlt.FeatureVector(float_features=action),
                )
            )
        self.reward_network.train()
        return reward_estimates.q_value.cpu().data.numpy()
