#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import List, Optional

import torch
from reagent.core.configuration import resolve_defaults
from reagent.core.types import CBInput
from reagent.gym.policies.policy import Policy
from reagent.models.disjoint_linucb_predictor import DisjointLinearRegressionUCB
from reagent.training.cb.base_trainer import BaseCBTrainerWithEval


logger = logging.getLogger(__name__)


class DisjointLinUCBTrainer(BaseCBTrainerWithEval):
    """
    The trainer for Disjoint LinUCB Contextual Bandit model.
    The model estimates a ridge regression (linear) and only supports dense features.

    Args:
        policy: The policy to be trained. Its scorer has to be DisjointLinearRegressionUCB
    """

    @resolve_defaults
    def __init__(
        self,
        policy: Policy,
        automatic_optimization: bool = False,  # turn off automatic_optimization because we are updating parameters manually
        *args,
        **kwargs,
    ):
        super().__init__(automatic_optimization=automatic_optimization, *args, **kwargs)
        assert isinstance(
            policy.scorer, DisjointLinearRegressionUCB
        ), "DisjointLinUCBTrainer requires the policy scorer to be DisjointLinearRegressionUCB"
        self.scorer = policy.scorer
        self.num_arms = policy.scorer.num_arms

    def configure_optimizers(self):
        # no optimizers bcs we update weights manually
        return None

    def update_params(
        self,
        arm_idx: int,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        weight: Optional[torch.Tensor] = None,
    ):
        """
        Update A and b for arm with index arm_idx
        Args:
            arm_idx: the index of the arm to be updated
            x: 2D tensor of shape (batch_size, dim)
            y: 2D tensor of shape (batch_size, 1)
            weight: 2D tensor of shape (batch_size, 1)
        """
        # weight is number of observations represented by each entry
        if weight is None:
            weight = torch.ones_like(torch.tensor(y))
        weight = weight.float()

        self.scorer.cur_num_obs[arm_idx] += torch.tensor(y).shape[0]

        self.scorer.cur_A[arm_idx] += torch.matmul(
            x.t(), x * weight
        )  # dim (DA*DC, DA*DC)
        self.scorer.cur_b[arm_idx] += torch.matmul(
            x.t(), y * weight
        ).squeeze()  # dim (DA*DC,)

    # pyre-ignore
    def _check_input(self, batch: List[CBInput]):
        # TODO: check later with train_script for batch's dataset info
        assert len(batch) == self.num_arms
        for sub_batch in batch:
            assert sub_batch.context_arm_features.ndim == 2
            assert sub_batch.reward is not None

    # pyre-fixme[14]: `cb_training_step` overrides method defined in `BaseCBTrainerWithEval`
    #  inconsistently.
    def cb_training_step(
        self, batch: List[CBInput], batch_idx: int, optimizer_idx: int = 0
    ):
        """
        each element in batch is a sub-batch of data for that arm
        """
        for arm_idx in range(self.num_arms):
            sub_batch = batch[arm_idx]
            self.update_params(
                arm_idx,
                sub_batch.context_arm_features,
                sub_batch.reward,
                sub_batch.weight,
            )

    def apply_discounting_multiplier(self):
        self.scorer.b *= self.scorer.gamma
        self.scorer.A *= self.scorer.gamma

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        # at the end of the training epoch calculate the coefficients
        self.scorer._estimate_coefs()
        # apply discount factor here so that next round it's already discounted
        # self.A is V in D-LinUCB paper https://arxiv.org/pdf/1909.09146.pdf
        # This is a simplified version of D-LinUCB, we calculate A = \sum \gamma^t xx^T
        # to discount the old data. See N2441818 for why we do this.
        self.apply_discounting_multiplier()
