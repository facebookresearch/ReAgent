#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

import reagent.types as rlt
import torch
import torch.nn as nn
import torch.nn.functional as F
from reagent.core.dataclasses import field
from reagent.model_utils.seq2slate_utils import Seq2SlateMode
from reagent.models.seq2slate import Seq2SlateTransformerNet
from reagent.optimizer.union import Optimizer__Union
from reagent.parameters import Seq2SlateParameters
from reagent.training.trainer import Trainer


logger = logging.getLogger(__name__)


class Seq2SlateTeacherForcingTrainer(Trainer):
    """
    Seq2Slate learned in a teach-forcing fashion (only used if the
    the ground-truth sequences are available)
    """

    def __init__(
        self,
        seq2slate_net: Seq2SlateTransformerNet,
        parameters: Seq2SlateParameters,
        minibatch_size: int,
        use_gpu: bool = False,
        policy_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        policy_gradient_interval: int = 1,
        print_interval: int = 100,
    ) -> None:
        self.parameters = parameters
        self.use_gpu = use_gpu
        self.policy_gradient_interval = policy_gradient_interval
        self.print_interval = print_interval
        self.seq2slate_net = seq2slate_net
        self.minibatch_size = minibatch_size
        self.minibatch = 0
        self.optimizer = policy_optimizer.make_optimizer(
            self.seq2slate_net.parameters()
        )
        self.optimizer.zero_grad()
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

    def warm_start_components(self):
        components = ["seq2slate_net"]
        return components

    def train(self, training_batch: rlt.PreprocessedTrainingBatch):
        assert type(training_batch) is rlt.PreprocessedTrainingBatch
        training_input = training_batch.training_input
        assert isinstance(training_input, rlt.PreprocessedRankingInput)
        self.minibatch += 1

        log_probs = self.seq2slate_net(
            training_input, mode=Seq2SlateMode.PER_SYMBOL_LOG_PROB_DIST_MODE
        ).log_probs
        assert log_probs.requires_grad

        assert training_input.optim_tgt_out_idx is not None
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got
        #  `Optional[torch.Tensor]`.
        labels = self._transform_label(training_input.optim_tgt_out_idx)
        assert not labels.requires_grad
        loss = self.kl_div_loss(log_probs, labels)

        loss.backward()
        if self.minibatch % self.policy_gradient_interval == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        loss = loss.detach().cpu().numpy()
        log_probs = log_probs.detach()
        if self.minibatch % self.print_interval == 0:
            logger.info(f"{self.minibatch} batch: loss={loss}")

        return log_probs, loss

    def _transform_label(self, optim_tgt_out_idx: torch.Tensor):
        label_size = self.seq2slate_net.max_src_seq_len + 2
        label = F.one_hot(optim_tgt_out_idx, label_size)
        return label.float()
