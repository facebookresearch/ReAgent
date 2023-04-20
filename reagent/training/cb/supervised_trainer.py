#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from enum import Enum

import torch
from reagent.core.types import CBInput
from reagent.gym.policies.policy import Policy
from reagent.training.cb.base_trainer import BaseCBTrainerWithEval

logger = logging.getLogger(__name__)


class LossTypes(Enum):
    mse = torch.nn.MSELoss
    mae = torch.nn.L1Loss
    cross_entropy = torch.nn.CrossEntropyLoss


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
        self.loss = LossTypes[loss_type].value()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def cb_training_step(
        self, batch: CBInput, batch_idx: int, optimizer_idx: int = 0
    ) -> torch.Tensor:
        assert batch.reward is not None  # to satisfy Pyre

        # compute the NN loss
        scores = self.scorer(batch.features_of_chosen_arm)
        return self.loss(scores, batch.reward.squeeze(-1))
