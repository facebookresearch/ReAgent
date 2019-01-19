#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Optional, Tuple

import ml.rl.types as rlt
import torch
import torch.nn.functional as F
from ml.rl.thrift.core.ttypes import DiscreteActionModelParameters
from ml.rl.training.dqn_trainer_base import DQNTrainerBase
from ml.rl.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


class _DQNTrainer(DQNTrainerBase):
    def __init__(
        self,
        q_network,
        q_network_target,
        reward_network,
        parameters: DiscreteActionModelParameters,
    ) -> None:
        self.double_q_learning = parameters.rainbow.double_q_learning
        self.minibatch_size = parameters.training.minibatch_size
        self._actions = parameters.actions if parameters.actions is not None else []

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

        self.reward_boosts = torch.zeros([1, len(self._actions)]).type(self.dtype)
        if parameters.rl.reward_boost is not None:
            for k in parameters.rl.reward_boost.keys():
                i = self._actions.index(k)
                self.reward_boosts[0, i] = parameters.rl.reward_boost[k]

    def get_detached_q_values(
        self, state
    ) -> Tuple[rlt.AllActionQValues, Optional[rlt.AllActionQValues]]:
        """ Gets the q values from the model and target networks """
        with torch.no_grad():
            input = rlt.StateInput(state=state)
            q_values = self.q_network(input)
            if self.double_q_learning:
                q_values_target = self.q_network_target(input)
            else:
                q_values_target = None
        return q_values, q_values_target

    def train(self, training_batch):
        if isinstance(training_batch, TrainingDataPage):
            training_batch = training_batch.as_discrete_sarsa_training_batch()

        learning_input = training_batch.training_input
        # Apply reward boost if specified
        reward_boosts = torch.sum(
            learning_input.action * self.reward_boosts, dim=1, keepdim=True
        )
        boosted_rewards = learning_input.reward + reward_boosts

        self.minibatch += 1
        rewards = boosted_rewards
        discount_tensor = torch.full_like(rewards, self.gamma)
        not_done_mask = learning_input.not_terminal

        if self.use_seq_num_diff_as_time_diff:
            # TODO: Implement this in another diff
            logger.warning(
                "_dqn_trainer has not implemented use_seq_num_diff_as_time_diff feature"
            )
            pass

        all_next_q_values, all_next_q_values_target = self.get_detached_q_values(
            learning_input.next_state
        )

        if self.maxq_learning:
            # Compute max a' Q(s', a') over all possible actions using target network
            next_q_values, max_q_action_idxs = self.get_max_q_values_with_target(
                all_next_q_values.q_values,
                all_next_q_values_target.q_values if self.double_q_learning else None,
                learning_input.possible_next_actions_mask.float(),
            )
        else:
            # SARSA
            next_q_values, max_q_action_idxs = self.get_max_q_values_with_target(
                all_next_q_values.q_values,
                all_next_q_values_target.q_values if self.double_q_learning else None,
                learning_input.next_action,
            )

        filtered_next_q_vals = next_q_values * not_done_mask.float()

        if self.minibatch < self.reward_burnin:
            target_q_values = rewards
        else:
            target_q_values = rewards + (discount_tensor * filtered_next_q_vals)

        # Get Q-value of action taken
        current_state = rlt.StateInput(state=learning_input.state)
        all_q_values = self.q_network(current_state).q_values
        self.all_action_scores = all_q_values.detach()
        q_values = torch.sum(all_q_values * learning_input.action, 1, keepdim=True)

        loss = self.q_network_loss(q_values, target_q_values)
        self.loss = loss.detach()

        self.q_network_optimizer.zero_grad()
        loss.backward()
        if self.gradient_handler:
            self.gradient_handler(self.q_network.parameters())
        self.q_network_optimizer.step()

        if self.minibatch < self.reward_burnin:
            # Reward burnin: force target network
            self._soft_update(self.q_network, self.q_network_target, 1.0)
        else:
            # Use the soft update rule to update target network
            self._soft_update(self.q_network, self.q_network_target, self.tau)

        # get reward estimates
        reward_estimates = self.reward_network(current_state).q_values
        reward_loss = F.mse_loss(reward_estimates, rewards)
        self.reward_network_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_network_optimizer.step()

        self.loss_reporter.report(
            td_loss=self.loss,
            reward_loss=reward_loss,
            model_values_on_logged_actions=q_values,
        )
