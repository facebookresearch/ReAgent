#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

import torch
from reagent.core.types import CBInput
from reagent.gym.policies.policy import Policy
from reagent.training.cb.base_trainer import BaseCBTrainerWithEval

logger = logging.getLogger(__name__)


LOSS_TYPES = {
    "mse": torch.nn.functional.mse_loss,
    "mae": torch.nn.functional.l1_loss,
    "cross_entropy": torch.nn.functional.binary_cross_entropy,
}


class SupervisedTrainer(BaseCBTrainerWithEval):
    """
    The trainer with a supervised learning loss. Supports Cross-Entropy, MSE and MAE losses.

    Args:
        policy: The policy to be trained.
    """

    def __init__(
        self,
        policy: Policy,
        loss_type: str = "mse",  # one of the LossTypes names
        lr: float = 1e-3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scorer = policy.scorer
        self.lr = lr
        self.loss = LOSS_TYPES[loss_type]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def cb_training_step(
        self, batch: CBInput, batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:
        assert batch.reward is not None  # to satisfy Pyre

        # compute the NN loss
        model_output = self.scorer(batch.features_of_chosen_arm)
        pred_reward = model_output["pred_reward"]

        # The supervised learning model outputs predicted reward with no uncertainty(uncertainty=ucb_alpha*pred_sigma).
        if batch.weight is not None:
            # weighted average loss
            losses = self.loss(pred_reward, batch.reward.squeeze(-1), reduction="none")
            return (losses * batch.weight.squeeze(-1)).sum() / batch.weight.sum()
        else:
            # non-weighted average loss
            return self.loss(pred_reward, batch.reward.squeeze(-1), reduction="mean")
