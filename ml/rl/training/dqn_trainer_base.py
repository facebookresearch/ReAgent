#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import torch
from ml.rl.training.rl_trainer_pytorch import RLTrainer


logger = logging.getLogger(__name__)


class DQNTrainerBase(RLTrainer):
    def get_max_q_values(self, q_values, possible_actions_mask):
        """
        Used in Q-learning update.

        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.

        :param possible_actions_mask: Numpy array with shape (batch_size, action_dim).
            possible_actions[i][j] = 1 iff the agent can take action j from
            state i.

        :param double_q_learning: bool to use double q-learning
        """

        # The parametric DQN can create flattened q values so we reshape here.
        q_values = q_values.reshape(possible_actions_mask.shape)

        # Set q-values of impossible actions to a very large negative number.
        inverse_pna = 1 - possible_actions_mask
        impossible_action_penalty = self.ACTION_NOT_POSSIBLE_VAL * inverse_pna
        q_values = q_values + impossible_action_penalty
        max_q_values, max_indicies = torch.max(q_values, dim=1, keepdim=True)
        return max_q_values, max_indicies

    def get_max_q_values_with_target(
        self, q_values, q_values_target, possible_actions_mask
    ):
        """
        Used in Q-learning update.

        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.

        :param possible_actions_mask: Numpy array with shape (batch_size, action_dim).
            possible_actions[i][j] = 1 iff the agent can take action j from
            state i.

        :param double_q_learning: bool to use double q-learning
        """

        # The parametric DQN can create flattened q values so we reshape here.
        q_values = q_values.reshape(possible_actions_mask.shape)
        q_values_target = q_values_target.reshape(possible_actions_mask.shape)

        if self.double_q_learning:
            # Set q-values of impossible actions to a very large negative number.
            inverse_pna = 1 - possible_actions_mask
            impossible_action_penalty = self.ACTION_NOT_POSSIBLE_VAL * inverse_pna
            q_values = q_values + impossible_action_penalty
            # Select max_q action after scoring with online network
            max_q_values, max_indicies = torch.max(q_values, dim=1, keepdim=True)
            # Use q_values from target network for max_q action from online q_network
            # to decouble selection & scoring, preventing overestimation of q-values
            max_q_values_target = torch.gather(q_values_target, 1, max_indicies)
            return max_q_values_target, max_indicies
        else:
            return self.get_max_q_values(q_values_target, possible_actions_mask)

    def boost_rewards(
        self, rewards: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        # Apply reward boost if specified
        reward_boosts = torch.sum(
            actions.float() * self.reward_boosts, dim=1, keepdim=True
        )
        return rewards + reward_boosts
