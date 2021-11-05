#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import Optional

import torch
from reagent.core.configuration import resolve_defaults
from reagent.core.types import CBInput
from reagent.gym.policies.policy import Policy
from reagent.models.linear_regression import LinearRegressionUCB
from reagent.training.reagent_lightning_module import ReAgentLightningModule


logger = logging.getLogger(__name__)


def _get_chosen_action_features(
    all_action_features: torch.Tensor, chosen_actions: torch.Tensor
) -> torch.Tensor:
    """
    Pick the features for chosen actions out of a tensor with features of all actions

    Args:
        all_action_features: 3D Tensor of shape (batch_size, num_actions, action_dim) with
            features of all available actions.
        chosen_actions: 2D Tensor of shape (batch_size, 1) with dtype long. For each observation
            it holds the index of the chosen action.
    Returns:
        A 2D Tensor of shape (batch_size, action_dim) with features of chosen actions.
    """
    assert all_action_features.ndim == 3
    return torch.gather(
        all_action_features,
        1,
        chosen_actions.unsqueeze(-1).expand(-1, 1, all_action_features.shape[2]),
    ).squeeze(1)


class LinUCBTrainer(ReAgentLightningModule):
    """
    The trainer for LinUCB Contextual Bandit model.
    The model estimates a ridge regression (linear) and only supports dense features.
    The actions are assumed to be one of:
        - Fixed actions. The same (have the same semantic meaning) actions across all contexts.
            If actions are fixed, they can't have features associated with them.
        - Feature actions. We can have different number and identities of actions in each
            context. The actions must have features to represent their semantic meaning.
    Reference: https://arxiv.org/pdf/1003.0146.pdf

    Args:
        policy: The policy to be trained. Its scorer has to be LinearRegressionUCB
        num_actions: The number of actions. If num_actions==-1, the actions are assumed to be feature actions,
            otherwise they are assumed to be fixed actions.
        use_interaction_features: If True,
    """

    @resolve_defaults
    def __init__(
        self,
        policy: Policy,
        num_actions: int = -1,
        use_interaction_features: bool = True,
    ):
        # turn off automatic_optimization because we are updating parameters manually
        super().__init__(automatic_optimization=False)
        assert isinstance(
            policy.scorer, LinearRegressionUCB
        ), "LinUCBTrainer requires the policy scorer to be LinearRegressionUCB"
        self.scorer = policy.scorer
        if num_actions == -1:
            self.fixed_actions = False
        else:
            assert num_actions > 1, "num_actions has to be an integer >1"
            self.fixed_actions = True
        self.num_actions = num_actions
        self.use_interaction_features = use_interaction_features

    def configure_optimizers(self):
        # no optimizers bcs we update weights manually
        return None

    def update_params(
        self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: 2D tensor of shape (batch_size, dim)
            y: 2D tensor of shape (batch_size, 1)
            weight: 2D tensor of shape (batch_size, 1)
        """
        # weight is number of observations represented by each entry
        if weight is None:
            weight = torch.ones_like(y)
        self.scorer.A += torch.matmul(x.t(), x * weight)  # dim (DA*DC, DA*DC)
        self.scorer.b += torch.matmul(x.t(), y * weight).squeeze()  # dim (DA*DC,)

    def _check_input(self, batch: CBInput):
        assert batch.context_action_features.ndim == 3
        assert batch.reward is not None
        assert batch.action is not None
        assert len(batch.action) == len(batch.reward)
        assert len(batch.action) == batch.context_action_features.shape[0]

    def training_step(self, batch: CBInput, batch_idx: int, optimizer_idx: int = 0):
        self._check_input(batch)
        assert batch.action is not None  # to satisfy Pyre
        x = _get_chosen_action_features(batch.context_action_features, batch.action)

        # update parameters
        assert batch.reward is not None  # to satisfy Pyre
        self.update_params(x, batch.reward, batch.weight)
