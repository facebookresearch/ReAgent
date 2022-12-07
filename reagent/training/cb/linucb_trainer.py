#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import Optional

import torch
from reagent.core.configuration import resolve_defaults
from reagent.core.types import CBInput
from reagent.gym.policies.policy import Policy
from reagent.models.linear_regression import LinearRegressionUCB
from reagent.training.cb.base_trainer import BaseCBTrainerWithEval


logger = logging.getLogger(__name__)


def _get_chosen_arm_features(
    all_arm_features: torch.Tensor, chosen_arms: torch.Tensor
) -> torch.Tensor:
    """
    Pick the features for chosen arms out of a tensor with features of all arms

    Args:
        all_arm_features: 3D Tensor of shape (batch_size, num_arms, arm_dim) with
            features of all available arms.
        chosen_arms: 2D Tensor of shape (batch_size, 1) with dtype long. For each observation
            it holds the index of the chosen arm.
    Returns:
        A 2D Tensor of shape (batch_size, arm_dim) with features of chosen arms.
    """
    assert all_arm_features.ndim == 3
    return torch.gather(
        all_arm_features,
        1,
        chosen_arms.unsqueeze(-1).expand(-1, 1, all_arm_features.shape[2]),
    ).squeeze(1)


class LinUCBTrainer(BaseCBTrainerWithEval):
    """
    The trainer for LinUCB Contextual Bandit model.
    The model estimates a ridge regression (linear) and only supports dense features.
    The arms can be one of 2 options (specified in FbContBanditBatchPreprocessor):
        - Fixed arms. The same (have the same semantic meaning) arms across all contexts.
            If arms are fixed, they can't have features associated with them. Used if
            `arm_normalization_data` not specified in FbContBanditBatchPreprocessor
        - Feature arms. We can have different number and identities of arms in each
            context. The arms must have features to represent their semantic meaning.
            Used if `arm_normalization_data` is specified in FbContBanditBatchPreprocessor
            and arm_features column is non-empty
    Reference: https://arxiv.org/pdf/1003.0146.pdf

    Args:
        policy: The policy to be trained. Its scorer has to be LinearRegressionUCB
    """

    @resolve_defaults
    def __init__(
        self,
        policy: Policy,
        automatic_optimization: bool = False,  # turn off automatic_optimization because we are updating parameters manually,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._automatic_optimization = automatic_optimization
        assert isinstance(
            policy.scorer, LinearRegressionUCB
        ), "LinUCBTrainer requires the policy scorer to be LinearRegressionUCB"
        self.scorer = policy.scorer

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
        weight = weight.float()

        self.scorer.cur_A += torch.matmul(x.t(), x * weight)  # dim (DA*DC, DA*DC)
        self.scorer.cur_b += torch.matmul(x.t(), y * weight).squeeze()  # dim (DA*DC,)
        self.scorer.cur_num_obs += y.shape[0]
        self.scorer.cur_sum_weight += weight.sum()

    def _check_input(self, batch: CBInput):
        assert batch.context_arm_features.ndim == 3
        assert batch.reward is not None
        assert batch.action is not None
        assert len(batch.action) == len(batch.reward)
        assert len(batch.action) == batch.context_arm_features.shape[0]

    def cb_training_step(self, batch: CBInput, batch_idx: int, optimizer_idx: int = 0):
        self._check_input(batch)
        assert batch.action is not None  # to satisfy Pyre
        x = _get_chosen_arm_features(batch.context_arm_features, batch.action)

        # update parameters
        assert batch.reward is not None  # to satisfy Pyre
        self.update_params(x, batch.reward, batch.weight)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        # at the end of the training epoch calculate the coefficients
        self.scorer._calculate_coefs()
        self.scorer.A *= self.scorer.gamma
        self.scorer.b *= self.scorer.gamma
