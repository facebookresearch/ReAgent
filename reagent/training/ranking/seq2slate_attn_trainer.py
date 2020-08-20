#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

import reagent.core.types as rlt
import torch
import torch.nn as nn
from reagent.core.dataclasses import field
from reagent.models.seq2slate import Seq2SlateMode, Seq2SlateTransformerNet
from reagent.optimizer.union import Optimizer__Union
from reagent.parameters import TransformerParameters
from reagent.training.trainer import Trainer


logger = logging.getLogger(__name__)


class Seq2SlatePairwiseAttnTrainer(Trainer):
    """
    Seq2Slate without a decoder learned in a supervised learning fashion (
    https://arxiv.org/pdf/1904.06813.pdf )
    """

    def __init__(
        self,
        seq2slate_net: Seq2SlateTransformerNet,
        minibatch_size: int = 1024,
        reporter=None,
        use_gpu: bool = False,
        policy_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
    ) -> None:
        self.reporter = reporter
        self.use_gpu = use_gpu
        self.seq2slate_net = seq2slate_net
        self.minibatch_size = minibatch_size
        self.minibatch = 0
        self.optimizer = policy_optimizer.make_optimizer(
            self.seq2slate_net.parameters()
        )
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def warm_start_components(self):
        components = ["seq2slate_net"]
        return components

    def train(self, training_batch: rlt.PreprocessedTrainingBatch):
        assert type(training_batch) is rlt.PreprocessedTrainingBatch
        training_input = training_batch.training_input
        assert isinstance(training_input, rlt.PreprocessedRankingInput)

        # shape: batch_size, tgt_seq_len
        encoder_scores = self.seq2slate_net(
            training_input, mode=Seq2SlateMode.ENCODER_SCORE_MODE
        ).encoder_scores
        assert encoder_scores.requires_grad

        loss = self.kl_loss(
            self.log_softmax(encoder_scores), training_input.position_reward
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss = loss.detach()
        self.minibatch += 1

        self.reporter.report(cross_entropy_loss=loss)

        return {"cross_entropy_loss": loss}
