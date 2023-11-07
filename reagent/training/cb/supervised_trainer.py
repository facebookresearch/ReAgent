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
        loss_type: str = "mse",  # one of the LOSS_TYPES names
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scorer = policy.scorer
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss = LOSS_TYPES[loss_type]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def cb_training_step(
        self, batch: CBInput, batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:
        assert batch.label is not None  # to satisfy Pyre

        # compute the NN loss
        model_output = self.scorer(batch.features_of_chosen_arm)
        pred_label = model_output["pred_label"]

        # The supervised learning model outputs predicted label with no uncertainty(uncertainty=ucb_alpha*pred_sigma).
        # weighted average loss
        losses = self.loss(pred_label, batch.label.squeeze(-1), reduction="none")
        weight = batch.effective_weight
        return (losses * weight.squeeze(-1)).sum() / losses.shape[0]
