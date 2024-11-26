#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging

import reagent.core.types as rlt
import torch
from reagent.core.dataclasses import field
from reagent.models.base import ModelBase
from reagent.optimizer.union import Optimizer__Union
from reagent.training.reagent_lightning_module import ReAgentLightningModule

logger = logging.getLogger(__name__)


class BehavioralCloningTrainer(ReAgentLightningModule):
    def __init__(
        self,
        bc_net: ModelBase,
        optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
    ) -> None:
        super().__init__()
        self.bc_net = bc_net
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        self.optimizer = optimizer

    def configure_optimizers(self):
        optimizers = []
        optimizers.append(
            self.optimizer.make_optimizer_scheduler(self.bc_net.parameters())
        )
        return optimizers

    def _get_masked_logits(self, batch: rlt.BehavioralCloningModelInput):
        logits = self.bc_net(
            batch.state, possible_actions_mask=batch.possible_actions_mask
        )
        return logits

    def train_step_gen(
        self, training_batch: rlt.BehavioralCloningModelInput, batch_idx: int
    ):
        self._check_input(training_batch)
        labels = training_batch.action
        logits_masked = self._get_masked_logits(training_batch)
        assert labels.ndim == logits_masked.ndim == 2
        assert labels.shape[0] == logits_masked.shape[0]
        _, integer_labels = labels.max(dim=0)
        loss = self.loss_fn(logits_masked, integer_labels)
        detached_loss = loss.detach().cpu()
        self.reporter.log(loss=detached_loss)
        yield loss

    # pyre-ignore inconsistent override because lightning doesn't use types
    def validation_step(self, batch: rlt.BehavioralCloningModelInput, batch_idx: int):
        self._check_input(batch)
        logits_masked = self._get_masked_logits(batch)
        labels = batch.action
        assert labels.ndim == logits_masked.ndim == 2
        assert labels.shape[0] == logits_masked.shape[0]
        _, integer_labels = labels.max(dim=0)
        loss = self.loss_fn(logits_masked, integer_labels)
        detached_loss = loss.detach().cpu()
        return detached_loss

    def _check_input(self, training_batch: rlt.BehavioralCloningModelInput):
        assert isinstance(training_batch, rlt.BehavioralCloningModelInput)
        labels = training_batch.action
        if len(labels.shape) > 1 and labels.shape[0] > 1:  # check one hot label
            pass
        else:
            raise TypeError(
                "label tensor format or dimension does not match loss function"
            )
        assert torch.all(
            # pyre-fixme[58]: `*` is not supported for operand types `Tensor` and
            #  `Optional[Tensor]`.
            training_batch.action * training_batch.possible_actions_mask
            == training_batch.action
        )  # check all labels are not masked out
