#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from enum import Enum
from typing import Optional

import numpy as np
import reagent.core.types as rlt
import torch
from reagent.core.dataclasses import field
from reagent.models.base import ModelBase
from reagent.optimizer.union import Optimizer__Union
from reagent.training.reagent_lightning_module import ReAgentLightningModule


logger = logging.getLogger(__name__)


class LossFunction(Enum):
    MSE = "MSE_Loss"
    SmoothL1Loss = "SmoothL1_Loss"
    L1Loss = "L1_Loss"
    BCELoss = "BCE_Loss"


def _get_loss_function(
    loss_fn: LossFunction,
    reward_ignore_threshold: Optional[float],
    weighted_by_inverse_propensity: bool,
):
    reduction_type = "none"

    if loss_fn == LossFunction.MSE:
        torch_fn = torch.nn.MSELoss(reduction=reduction_type)
    elif loss_fn == LossFunction.SmoothL1Loss:
        torch_fn = torch.nn.SmoothL1Loss(reduction=reduction_type)
    elif loss_fn == LossFunction.L1Loss:
        torch_fn = torch.nn.L1Loss(reduction=reduction_type)
    elif loss_fn == LossFunction.BCELoss:
        torch_fn = torch.nn.BCELoss(reduction=reduction_type)

    def wrapper_loss_fn(pred, target, weight):
        loss = torch_fn(pred, target)

        if weighted_by_inverse_propensity:
            assert weight.shape == loss.shape
            loss = loss * weight

        # ignore abnormal reward only during training
        if pred.requires_grad and reward_ignore_threshold is not None:
            loss = loss[target <= reward_ignore_threshold]
            assert len(loss) > 0, (
                f"reward ignore threshold set too small. target={target}, "
                f"threshold={reward_ignore_threshold}"
            )

        return torch.mean(loss)

    return wrapper_loss_fn


class RewardNetTrainer(ReAgentLightningModule):
    def __init__(
        self,
        reward_net: ModelBase,
        optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        loss_type: LossFunction = LossFunction.MSE,
        reward_ignore_threshold: Optional[float] = None,
        weighted_by_inverse_propensity: bool = False,
    ) -> None:
        super().__init__()
        self.reward_net = reward_net
        self.optimizer = optimizer
        self.loss_type = loss_type
        self.reward_ignore_threshold = reward_ignore_threshold
        self.weighted_by_inverse_propensity = weighted_by_inverse_propensity
        self.loss_fn = _get_loss_function(
            loss_type, reward_ignore_threshold, weighted_by_inverse_propensity
        )

    def configure_optimizers(self):
        optimizers = []
        optimizers.append(
            self.optimizer.make_optimizer_scheduler(self.reward_net.parameters())
        )
        return optimizers

    def _get_sample_weight(self, batch: rlt.PreprocessedRankingInput):
        weight = None
        if self.weighted_by_inverse_propensity:
            if isinstance(batch, rlt.PreprocessedRankingInput):
                assert batch.tgt_out_probs is not None
                # pyre-fixme[58]: `/` is not supported for operand types `float` and
                #  `Optional[torch.Tensor]`.
                weight = 1.0 / batch.tgt_out_probs
            else:
                raise NotImplementedError(
                    f"Sampling weighting not implemented for {type(batch)}"
                )
        return weight

    def _get_target_reward(self, batch: rlt.PreprocessedRankingInput):
        if isinstance(batch, rlt.PreprocessedRankingInput):
            target_reward = batch.slate_reward
        else:
            target_reward = batch.reward
        assert target_reward is not None
        return target_reward

    # pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
    #  its type `no_grad` is not callable.
    @torch.no_grad()
    def _compute_unweighted_loss(
        self, predicted_reward: torch.Tensor, target_reward: torch.Tensor
    ):
        return self.loss_fn(predicted_reward, target_reward, weight=None)

    def train_step_gen(
        self, training_batch: rlt.PreprocessedRankingInput, batch_idx: int
    ):
        weight = self._get_sample_weight(training_batch)
        target_reward = self._get_target_reward(training_batch)
        predicted_reward = self.reward_net(training_batch).predicted_reward

        assert (
            predicted_reward.shape == target_reward.shape
            and len(target_reward.shape) == 2
            and target_reward.shape[1] == 1
        )
        loss = self.loss_fn(predicted_reward, target_reward, weight)

        detached_loss = loss.detach().cpu()
        self.reporter.log(loss=detached_loss)

        if weight is not None:
            unweighted_loss = self._compute_unweighted_loss(
                predicted_reward, target_reward
            )
            self.reporter.log(unweighted_loss=unweighted_loss)

        if self.all_batches_processed % 10 == 0:
            logger.info(
                f"{self.all_batches_processed}-th batch: "
                f"{self.loss_type}={detached_loss.item()}"
            )

        yield loss

    # pyre-ignore inconsistent override because lightning doesn't use types
    def validation_step(self, batch: rlt.PreprocessedRankingInput, batch_idx: int):
        reward = self._get_target_reward(batch)
        self.reporter.log(eval_rewards=reward.flatten().detach().cpu())

        pred_reward = self.reward_net(batch).predicted_reward
        self.reporter.log(eval_pred_rewards=pred_reward.flatten().detach().cpu())

        weight = self._get_sample_weight(batch)
        loss = self.loss_fn(pred_reward, reward, weight)

        detached_loss = loss.detach().cpu()
        self.reporter.log(eval_loss=detached_loss)

        if weight is not None:
            unweighted_loss = self._compute_unweighted_loss(pred_reward, reward)
            self.reporter.log(eval_unweighted_loss=unweighted_loss)

        return detached_loss.item()

    def validation_epoch_end(self, outputs):
        self.reporter.update_best_model(np.mean(outputs), self.reward_net)

    def warm_start_components(self):
        return ["reward_net"]
