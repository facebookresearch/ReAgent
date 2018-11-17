#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import ml.rl.types as rlt
import numpy as np
import torch
import torch.nn.functional as F
from ml.rl.caffe_utils import arange_expand
from ml.rl.thrift.core.ttypes import ContinuousActionModelParameters
from ml.rl.training.evaluator import BatchStatsForCPE
from ml.rl.training.rl_trainer_pytorch import RLTrainer


logger = logging.getLogger(__name__)


class _ParametricDQNTrainer(RLTrainer):
    def __init__(
        self,
        q_network,
        q_network_target,
        reward_network,
        parameters: ContinuousActionModelParameters,
    ) -> None:
        self.double_q_learning = parameters.rainbow.double_q_learning
        self.minibatch_size = parameters.training.minibatch_size

        RLTrainer.__init__(
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

    def get_max_q_values(
        self, tiled_next_state, possible_next_actions, double_q_learning
    ):
        """
        :param double_q_learning: bool to use double q-learning
        """

        lengths = possible_next_actions.lengths
        row_nums = np.arange(len(lengths))
        row_idxs = np.repeat(row_nums, lengths.cpu().numpy())
        col_idxs = arange_expand(lengths).cpu().numpy()

        dense_idxs = torch.tensor(
            (row_idxs, col_idxs), device=lengths.device, dtype=torch.int64
        )

        q_network_input = rlt.StateAction(
            state=tiled_next_state, action=possible_next_actions.actions
        )
        if double_q_learning:
            q_values = self.q_network(q_network_input).q_value.squeeze().detach()
            q_values_target = (
                self.q_network_target(q_network_input).q_value.squeeze().detach()
            )
        else:
            q_values = self.q_network_target(q_network_input).q_value.squeeze().detach()

        dense_dim = [len(lengths), max(lengths)]
        # Add specific fingerprint to q-values so that after sparse -> dense we can
        # subtract the fingerprint to identify the 0's added in sparse -> dense
        q_values.add_(self.FINGERPRINT)
        sparse_q = torch.sparse_coo_tensor(dense_idxs, q_values, dense_dim)
        dense_q = sparse_q.to_dense()
        dense_q.add_(self.FINGERPRINT * -1)
        dense_q[dense_q == self.FINGERPRINT * -1] = self.ACTION_NOT_POSSIBLE_VAL
        max_q_values, max_indexes = torch.max(dense_q, dim=1)

        if double_q_learning:
            sparse_q_target = torch.sparse_coo_tensor(
                dense_idxs, q_values_target, dense_dim
            )
            dense_q_values_target = sparse_q_target.to_dense()
            max_q_values = torch.gather(
                dense_q_values_target, 1, max_indexes.unsqueeze(1)
            )

        return max_q_values.squeeze()

    def get_next_action_q_values(self, state, action):
        return self.q_network_target(
            rlt.StateAction(state=state, action=action)
        ).q_value

    def train(self, training_batch, evaluator=None) -> None:
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
            # Compute max a' Q(s', a') over all possible actions using target network
            next_q_values = self.get_max_q_values(
                learning_input.tiled_next_state,
                learning_input.possible_next_actions,
                self.double_q_learning,
            )
        else:
            # SARSA
            next_q_values = self.get_next_action_q_values(
                learning_input.next_state, learning_input.next_action
            )

        filtered_max_q_vals = next_q_values.reshape(-1, 1) * not_done_mask

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
            td_loss=float(self.loss), reward_loss=float(reward_loss)
        )

        if evaluator is not None:
            cpe_stats = BatchStatsForCPE(
                model_values_on_logged_actions=self.all_action_scores
            )
            evaluator.report(cpe_stats)
