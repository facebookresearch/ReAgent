#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Optional

import numpy as np
import reagent.core.types as rlt
import torch
from reagent.core.dataclasses import field
from reagent.models.base import ModelBase
from reagent.optimizer.union import Optimizer__Union
from reagent.training.reagent_lightning_module import ReAgentLightningModule
from reagent.training.reward_network_trainer import _get_loss_function, LossFunction

logger = logging.getLogger(__name__)


class BanditRewardNetTrainer(ReAgentLightningModule):
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

    def _get_sample_weight(self, batch: rlt.BanditRewardModelInput):
        weight = None
        if self.weighted_by_inverse_propensity:
            assert batch.action_prob is not None
            weight = 1.0 / batch.action_prob
        return weight

    def _get_predicted_reward(self, batch: rlt.BanditRewardModelInput):
        model_rewards_all_actions = self.reward_net(batch.state)
        logged_action_idxs = torch.argmax(batch.action, dim=1, keepdim=True)
        predicted_reward = model_rewards_all_actions.gather(1, logged_action_idxs)
        return predicted_reward

    @torch.no_grad()
    def _compute_unweighted_loss(
        self, predicted_reward: torch.Tensor, target_reward: torch.Tensor
    ):
        return self.loss_fn(
            predicted_reward, target_reward, weight=torch.ones_like(predicted_reward)
        )

    def train_step_gen(
        self, training_batch: rlt.BanditRewardModelInput, batch_idx: int
    ):
        weight = self._get_sample_weight(training_batch)
        target_reward = training_batch.reward
        predicted_reward = self._get_predicted_reward(training_batch)

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
    def validation_step(self, batch: rlt.BanditRewardModelInput, batch_idx: int):
        if self._training_batch_type and isinstance(batch, dict):
            batch = self._training_batch_type.from_dict(batch)

        reward = batch.reward
        self.reporter.log(eval_rewards=reward.flatten().detach().cpu())

        pred_reward = self._get_predicted_reward(batch)
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
