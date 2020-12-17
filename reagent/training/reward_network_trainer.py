#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from enum import Enum
from typing import Optional

import reagent.types as rlt
import torch
from reagent.core.dataclasses import field
from reagent.models.base import ModelBase
from reagent.optimizer.union import Optimizer__Union
from reagent.training.trainer import Trainer


logger = logging.getLogger(__name__)


class LossFunction(Enum):
    MSE = "MSE_Loss"
    SmoothL1Loss = "SmoothL1_Loss"
    L1Loss = "L1_Loss"
    BCELoss = "BCE_Loss"


def _get_loss_function(loss_fn: LossFunction, reward_ignore_threshold):
    reduction_type = "mean"
    if reward_ignore_threshold is not None:
        reduction_type = "none"

    if loss_fn == LossFunction.MSE:
        torch_fn = torch.nn.MSELoss(reduction=reduction_type)
    elif loss_fn == LossFunction.SmoothL1Loss:
        torch_fn = torch.nn.SmoothL1Loss(reduction=reduction_type)
    elif loss_fn == LossFunction.L1Loss:
        torch_fn = torch.nn.L1Loss(reduction=reduction_type)
    elif loss_fn == LossFunction.BCELoss:
        torch_fn = torch.nn.BCELoss(reduction=reduction_type)

    if reward_ignore_threshold is None:
        return torch_fn

    def wrapper_loss_fn(pred, target):
        loss = torch_fn(pred, target)
        # ignore abnormal reward only during training
        if pred.requires_grad:
            loss = loss[target <= reward_ignore_threshold]
            assert len(loss) > 0, (
                f"reward ignore threshold set too small. target={target}, "
                f"threshold={reward_ignore_threshold}"
            )
        return torch.mean(loss)

    return wrapper_loss_fn


class RewardNetTrainer(Trainer):
    def __init__(
        self,
        reward_net: ModelBase,
        use_gpu: bool = False,
        minibatch_size: int = 1024,
        optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        loss_type: LossFunction = LossFunction.MSE,
        reward_ignore_threshold: Optional[float] = None,
    ) -> None:
        self.reward_net = reward_net
        self.use_gpu = use_gpu
        self.minibatch_size = minibatch_size
        self.minibatch = 0
        self.opt = optimizer.make_optimizer(self.reward_net.parameters())
        self.loss_type = loss_type
        self.loss_fn = _get_loss_function(loss_type, reward_ignore_threshold)
        self.reward_ignore_threshold = reward_ignore_threshold

    def train(self, training_batch: rlt.PreprocessedTrainingBatch):
        with torch.no_grad():
            training_input = training_batch.training_input
            if isinstance(training_input, rlt.PreprocessedRankingInput):
                target_reward = training_input.slate_reward
            else:
                target_reward = training_input.reward

        predicted_reward = self.reward_net(training_input).predicted_reward
        loss = self.loss_fn(predicted_reward, target_reward.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        loss = loss.detach()

        self.minibatch += 1
        if self.minibatch % 10 == 0:
            logger.info(f"{self.minibatch}-th batch: {self.loss_type}={loss}")

        return loss

    def warm_start_components(self):
        return ["reward_net"]
