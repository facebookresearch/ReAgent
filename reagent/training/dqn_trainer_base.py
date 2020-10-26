#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import torch
from reagent.torch_utils import masked_softmax
from reagent.training.rl_trainer_pytorch import RLTrainer


logger = logging.getLogger(__name__)


class DQNTrainerBase(RLTrainer):
    def get_max_q_values(self, q_values, possible_actions_mask):
        return self.get_max_q_values_with_target(
            q_values, q_values, possible_actions_mask
        )

    def get_max_q_values_with_target(
        self, q_values, q_values_target, possible_actions_mask
    ):
        """
        Used in Q-learning update.

        :param q_values: PyTorch tensor with shape (batch_size, state_dim). Each row
            contains the list of Q-values for each possible action in this state.

        :param q_values_target: PyTorch tensor with shape (batch_size, state_dim). Each row
            contains the list of Q-values from the target network
            for each possible action in this state.

        :param possible_actions_mask: PyTorch tensor with shape (batch_size, action_dim).
            possible_actions[i][j] = 1 iff the agent can take action j from
            state i.

        Returns a tensor of maximum Q-values for every state in the batch
            and also the index of the corresponding action. NOTE: looks like
            this index is only used for informational purposes only and does
            not affect any algorithms.

        """

        # The parametric DQN can create flattened q values so we reshape here.
        q_values = q_values.reshape(possible_actions_mask.shape)
        q_values_target = q_values_target.reshape(possible_actions_mask.shape)
        # Set q-values of impossible actions to a very large negative number.
        inverse_pna = 1 - possible_actions_mask
        impossible_action_penalty = self.ACTION_NOT_POSSIBLE_VAL * inverse_pna
        q_values = q_values + impossible_action_penalty

        max_q_values, max_indicies = torch.max(q_values, dim=1, keepdim=True)
        if self.double_q_learning:
            # Use indices of the max q_values from the online network to select q-values
            # from the target network. This prevents overestimation of q-values.
            # The torch.gather function selects the entry from each row that corresponds
            # to the max_index in that row.
            max_q_values_target = torch.gather(q_values_target, 1, max_indicies)
        else:
            max_q_values_target = max_q_values

        return max_q_values_target, max_indicies

    # pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
    #  its type `no_grad` is not callable.
    @torch.no_grad()
    def boost_rewards(
        self, rewards: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        # Apply reward boost if specified
        reward_boosts = torch.sum(
            # pyre-fixme[16]: `DQNTrainerBase` has no attribute `reward_boosts`.
            actions.float() * self.reward_boosts,
            dim=1,
            keepdim=True,
        )
        return rewards + reward_boosts
