#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import Optional

import reagent.types as rlt
import torch
import torch.nn as nn
import torch.nn.functional as F
from reagent.models.seq2slate import (
    BaselineNet,
    Seq2SlateMode,
    Seq2SlateTransformerModel,
    Seq2SlateTransformerNet,
)
from reagent.parameters import Seq2SlateTransformerParameters
from reagent.training.trainer import Trainer


logger = logging.getLogger(__name__)


class Seq2SlateDifferentiableRewardTrainer(Trainer):
    """
    Seq2Slate learned with differentiable reward (Section 3.2 in
    https://arxiv.org/pdf/1810.02019.pdf )
    """

    def __init__(
        self,
        seq2slate_net: Seq2SlateTransformerNet,
        parameters: Seq2SlateTransformerParameters,
        minibatch_size: int,
        baseline_net: Optional[BaselineNet] = None,
        use_gpu: bool = False,
    ) -> None:
        self.parameters = parameters
        self.use_gpu = use_gpu
        self.seq2slate_net = seq2slate_net
        self.baseline_net = baseline_net
        self.minibatch_size = minibatch_size
        self.minibatch = 0
        self.optimizer = torch.optim.Adam(
            self.seq2slate_net.parameters(),
            lr=self.parameters.transformer.learning_rate,
            amsgrad=True,
        )
        # TODO: T62269969 add baseline_net in training
        self.kl_div_loss = nn.KLDivLoss(reduction="none")

    def warm_start_components(self):
        components = ["seq2slate_net"]
        return components

    def train(self, training_batch: rlt.PreprocessedTrainingBatch):
        assert type(training_batch) is rlt.PreprocessedTrainingBatch
        training_input = training_batch.training_input
        assert isinstance(training_input, rlt.PreprocessedRankingInput)

        per_symbol_log_probs = self.seq2slate_net(
            training_input, mode=Seq2SlateMode.PER_SYMBOL_LOG_PROB_DIST_MODE
        ).log_probs
        per_seq_log_probs = Seq2SlateTransformerModel.per_symbol_to_per_seq_log_probs(
            per_symbol_log_probs, training_input.tgt_out_idx
        )
        assert per_symbol_log_probs.requires_grad and per_seq_log_probs.requires_grad
        # pyre-fixme[16]: `Optional` has no attribute `shape`.
        assert per_seq_log_probs.shape == training_input.tgt_out_probs.shape

        if not self.parameters.on_policy:
            importance_sampling = (
                torch.exp(per_seq_log_probs) / training_input.tgt_out_probs
            )
            if self.parameters.importance_sampling_clamp_max is not None:
                importance_sampling = torch.clamp(
                    importance_sampling,
                    0,
                    self.parameters.importance_sampling_clamp_max,
                )
        else:
            importance_sampling = (
                torch.exp(per_seq_log_probs) / torch.exp(per_seq_log_probs).detach()
            )
        assert importance_sampling.requires_grad

        # pyre-fixme[6]: Expected `Tensor` for 1st param but got
        #  `Optional[torch.Tensor]`.
        labels = self._transform_label(training_input.tgt_out_idx)
        assert not labels.requires_grad

        batch_size, max_tgt_seq_len = training_input.tgt_out_idx.shape
        # batch_loss shape: batch_size x max_tgt_seq_len
        batch_loss = (
            torch.sum(self.kl_div_loss(per_symbol_log_probs, labels), dim=2)
            * training_input.position_reward
        )
        # weighted_batch_loss shape: batch_size, 1
        weighted_batch_loss = torch.sum(
            1.0
            / torch.log(
                torch.arange(1, 1 + max_tgt_seq_len, device=batch_loss.device).float()
                + 1.0
            )
            * batch_loss,
            dim=1,
            keepdim=True,
        )
        loss = 1.0 / batch_size * torch.sum(importance_sampling * weighted_batch_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss = loss.detach().cpu().numpy()
        per_symbol_log_probs = per_symbol_log_probs.detach()
        self.minibatch += 1
        logger.info(f"{self.minibatch} batch: loss={loss}")

        return {"per_symbol_log_probs": per_symbol_log_probs, "sl": loss}

    def _transform_label(self, tgt_out_idx: torch.Tensor):
        label_size = self.seq2slate_net.max_src_seq_len + 2
        label = F.one_hot(tgt_out_idx, label_size)
        return label.float()
