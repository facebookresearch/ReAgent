#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Tuple

import ml.rl.types as rlt
import torch
import torch.nn.functional as F
from ml.rl.thrift.core.ttypes import ContinuousActionModelParameters
from ml.rl.training.dqn_trainer_base import DQNTrainerBase


logger = logging.getLogger(__name__)


class _ParametricDQNTrainer(DQNTrainerBase):
    def __init__(
        self,
        q_network,
        q_network_target,
        reward_network,
        parameters: ContinuousActionModelParameters,
    ) -> None:
        self.double_q_learning = parameters.rainbow.double_q_learning
        self.minibatch_size = parameters.training.minibatch_size

        DQNTrainerBase.__init__(
            self,
            parameters,
            use_gpu=False,
            additional_feature_types=None,
            gradient_handler=None,
        )

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

    def get_detached_q_values(self, state, action) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Gets the q values from the model and target networks """
        with torch.no_grad():
            state_action_pairs = rlt.StateAction(state=state, action=action)
            q_values = self.q_network(state_action_pairs)
            q_values_target = self.q_network_target(state_action_pairs)
        return q_values, q_values_target

    def train(self, training_batch) -> None:
        if hasattr(training_batch, "as_parametric_sarsa_training_batch"):
            training_batch = training_batch.as_parametric_sarsa_training_batch()

        learning_input = training_batch.training_input
        self.minibatch += 1

        reward = learning_input.reward
        discount_tensor = torch.full_like(reward, self.gamma)
        not_done_mask = learning_input.not_terminal

        if self.use_seq_num_diff_as_time_diff:
            # TODO: Implement this in another diff
            raise NotImplementedError

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
            # SARSA
            next_q_values, _ = self.get_detached_q_values(
                learning_input.next_state, learning_input.next_action
            )
            next_q_values = next_q_values.q_value

        filtered_max_q_vals = next_q_values * not_done_mask.float()

        if self.minibatch < self.reward_burnin:
            target_q_values = reward
        else:
            target_q_values = reward + (discount_tensor * filtered_max_q_vals)

        # Get Q-value of action taken
        current_state_action = rlt.StateAction(
            state=learning_input.state, action=learning_input.action
        )
        q_values = self.q_network(current_state_action).q_value
        self.all_action_scores = q_values.detach()

        value_loss = self.q_network_loss(q_values, target_q_values)
        self.loss = value_loss.detach()

        self.q_network_optimizer.zero_grad()
        value_loss.backward()
        if self.gradient_handler:
            self.gradient_handler(self.q_network.parameters())
        self.q_network_optimizer.step()

        # TODO: Maybe soft_update should belong to the target network
        if self.minibatch < self.reward_burnin:
            # Reward burnin: force target network
            self._soft_update(self.q_network, self.q_network_target, 1.0)
        else:
            # Use the soft update rule to update target network
            self._soft_update(self.q_network, self.q_network_target, self.tau)

        # get reward estimates
        reward_estimates = self.reward_network(current_state_action).q_value
        reward_loss = F.mse_loss(reward_estimates, reward)
        self.reward_network_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_network_optimizer.step()

        self.loss_reporter.report(
            td_loss=self.loss,
            reward_loss=reward_loss,
            model_values_on_logged_actions=self.all_action_scores,
        )
