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


class LinUCBTrainer(BaseCBTrainerWithEval):
    """
    The trainer for LinUCB Contextual Bandit model.
    The model estimates a ridge regression (linear) and only supports dense features.
    We can have different number and identities of arms in each observation. The arms must
            have features to represent their semantic meaning.
    Instead of keeping track of cumulative values of `A` and `b`, we keep track of the average
        (cumulative divided by total weight) values.
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
        super().__init__(automatic_optimization=automatic_optimization, *args, **kwargs)
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

        batch_sum_weight = weight.sum()
        self.scorer.cur_num_obs += y.shape[0]
        self.scorer.cur_sum_weight += batch_sum_weight
        # update average values of A and b using observations from the batch
        self.scorer.cur_avg_A = (
            self.scorer.cur_avg_A * (1 - batch_sum_weight / self.scorer.cur_sum_weight)
            + torch.matmul(x.t(), x * weight) / self.scorer.cur_sum_weight
        )  # dim (DA*DC, DA*DC)
        self.scorer.cur_avg_b = (
            self.scorer.cur_avg_b * (1 - batch_sum_weight / self.scorer.cur_sum_weight)
            + torch.matmul(x.t(), y * weight).squeeze() / self.scorer.cur_sum_weight
        )  # dim (DA*DC,)

    def cb_training_step(
        self, batch: CBInput, batch_idx: int, optimizer_idx: int = 0
    ) -> Optional[torch.Tensor]:
        assert batch.label is not None  # to satisfy Pyre
        assert batch.features_of_chosen_arm is not None  # to satisfy Pyre

        # update parameters
        self.update_params(
            batch.features_of_chosen_arm, batch.label, batch.effective_weight
        )

    def apply_discounting_multiplier(self):
        self.scorer.sum_weight *= self.scorer.gamma

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        # at the end of the training epoch calculate the coefficients
        self.scorer._calculate_coefs()
        # apply discounting factor to the total weight. the average `A` and `b` valuse remain the same
        self.apply_discounting_multiplier()
