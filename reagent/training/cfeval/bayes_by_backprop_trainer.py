#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging

import reagent.core.types as rlt
import torch
from reagent.training.cfeval.bandit_reward_network_trainer import BanditRewardNetTrainer

logger = logging.getLogger(__name__)


class BayesByBackpropTrainer(BanditRewardNetTrainer):
    def train_step_gen(
        self, training_batch: rlt.BanditRewardModelInput, batch_idx: int
    ):
        weight = self._get_sample_weight(training_batch)

        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        loss = self.reward_net.sample_elbo(
            torch.cat([training_batch.action, training_batch.state.float_features], 1),
            training_batch.reward,
            1,
        )
        # use (action, reward) to indicate (input,target)

        detached_loss = loss.detach().cpu()
        self.reporter.log(loss=detached_loss)

        if weight is not None:
            raise NotImplementedError  # TODO for integration in to RL framework
            # unweighted_loss = self._compute_unweighted_loss(
            #    predicted_reward, target_reward, training_batch
            # )
            # self.reporter.log(unweighted_loss=unweighted_loss)

        self.all_batches_processed += 1
        if self.all_batches_processed % 100 == 0:
            logger.info(
                f"{self.all_batches_processed}-th batch: Loss={detached_loss.item()}"
            )

        yield loss

    def validation_step(self, batch: rlt.BanditRewardModelInput, batch_idx: int):
        if self._training_batch_type and isinstance(batch, dict):
            batch = self._training_batch_type.from_dict(batch)

        weight = self._get_sample_weight(batch)
        # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
        loss = self.reward_net.sample_elbo(
            torch.cat([batch.action, batch.state.float_features], 1),
            batch.reward,
            1,
        )

        detached_loss = loss.detach().cpu()
        self.reporter.log(eval_loss=detached_loss)

        if weight is not None:
            raise NotImplementedError  # TODO for integration in to RL framework
            # unweighted_loss = self._compute_unweighted_loss(pred_reward, reward, batch)
            # self.reporter.log(eval_unweighted_loss=unweighted_loss)

        return detached_loss.item()
