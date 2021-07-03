#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

import reagent.core.types as rlt
import torch
import torch.nn as nn
from reagent.core.dataclasses import field
from reagent.core.parameters import TransformerParameters
from reagent.core.tracker import observable
from reagent.model_utils.seq2slate_utils import Seq2SlateMode
from reagent.models.seq2slate import Seq2SlateTransformerNet
from reagent.optimizer.union import Optimizer__Union
from reagent.training.loss_reporter import NoOpLossReporter
from reagent.training.trainer import Trainer


logger = logging.getLogger(__name__)


@observable(cross_entropy_loss=torch.Tensor)
class Seq2SlatePairwiseAttnTrainer(Trainer):
    """
    Seq2Slate without a decoder learned in a supervised learning fashion (
    https://arxiv.org/pdf/1904.06813.pdf )
    """

    def __init__(
        self,
        seq2slate_net: Seq2SlateTransformerNet,
        loss_reporter=None,
        use_gpu: bool = False,
        policy_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
    ) -> None:
        self.loss_reporter = loss_reporter
        self.use_gpu = use_gpu
        self.seq2slate_net = seq2slate_net
        self.minibatch = 0
        self.optimizer = policy_optimizer.make_optimizer_scheduler(
            self.seq2slate_net.parameters()
        )["optimizer"]
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        if self.loss_reporter is None:
            self.loss_reporter = NoOpLossReporter()

    def warm_start_components(self):
        components = ["seq2slate_net"]
        return components

    def train(self, training_batch: rlt.PreprocessedRankingInput):
        assert type(training_batch) is rlt.PreprocessedRankingInput

        # shape: batch_size, tgt_seq_len
        encoder_scores = self.seq2slate_net(
            training_batch, mode=Seq2SlateMode.ENCODER_SCORE_MODE
        ).encoder_scores
        assert encoder_scores.requires_grad

        loss = self.kl_loss(
            self.log_softmax(encoder_scores), training_batch.position_reward
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss = loss.detach()
        self.minibatch += 1

        # pyre-fixme[16]: `Seq2SlatePairwiseAttnTrainer` has no attribute
        #  `notify_observers`.
        self.notify_observers(cross_entropy_loss=loss)

        return {"cross_entropy_loss": loss}
