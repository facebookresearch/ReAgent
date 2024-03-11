#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
import logging

import numpy as np
import reagent.core.types as rlt
import torch
import torch.nn as nn
from reagent.core.dataclasses import field
from reagent.model_utils.seq2slate_utils import Seq2SlateMode
from reagent.models.seq2slate import Seq2SlateTransformerNet
from reagent.optimizer.union import Optimizer__Union
from reagent.training.reagent_lightning_module import ReAgentLightningModule
from sklearn.metrics import (
    average_precision_score,
    dcg_score,
    ndcg_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class Seq2SlatePairwiseAttnTrainer(ReAgentLightningModule):
    """
    Seq2Slate without a decoder learned in a supervised learning fashion (
    https://arxiv.org/pdf/1904.06813.pdf )
    """

    def __init__(
        self,
        seq2slate_net: Seq2SlateTransformerNet,
        slate_size: int,
        calc_cpe: bool,
        policy_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
    ) -> None:
        super().__init__()
        self.seq2slate_net = seq2slate_net
        self.slate_size = slate_size
        self.calc_cpe = calc_cpe
        self.policy_optimizer = policy_optimizer
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def configure_optimizers(self):
        optimizers = []
        optimizers.append(
            self.policy_optimizer.make_optimizer_scheduler(
                self.seq2slate_net.parameters()
            )
        )
        return optimizers

    def train_step_gen(
        self, training_batch: rlt.PreprocessedRankingInput, batch_idx: int
    ):
        assert type(training_batch) is rlt.PreprocessedRankingInput

        # shape: batch_size, tgt_seq_len
        encoder_scores = self.seq2slate_net(
            training_batch, mode=Seq2SlateMode.ENCODER_SCORE_MODE
        ).encoder_scores
        assert encoder_scores.requires_grad

        loss = self.kl_loss(
            self.log_softmax(encoder_scores), training_batch.position_reward
        )

        detached_loss = loss.detach().cpu()
        self.reporter.log(train_cross_entropy_loss=detached_loss)

        yield loss

    # pyre-ignore inconsistent override because lightning doesn't use types
    def validation_step(self, batch: rlt.PreprocessedRankingInput, batch_idx: int):
        # pyre-fixme[16]: `Optional` has no attribute `shape`.
        batch_size = batch.position_reward.shape[0]

        # shape: batch_size, tgt_seq_len
        encoder_scores = self.seq2slate_net(
            batch, mode=Seq2SlateMode.ENCODER_SCORE_MODE
        ).encoder_scores
        assert (
            encoder_scores.shape[1] == batch.position_reward.shape[1] == self.slate_size
        )
        ce_loss = self.kl_loss(
            self.log_softmax(encoder_scores), batch.position_reward
        ).item()

        if not self.calc_cpe:
            self.reporter.log(eval_cross_entropy_loss=ce_loss)
            return

        # shape: batch_size, tgt_seq_len
        ranking_output = self.seq2slate_net(
            batch, mode=Seq2SlateMode.RANK_MODE, greedy=True
        )
        # pyre-fixme[16]: `int` has no attribute `cpu`.
        ranked_idx = (ranking_output.ranked_tgt_out_idx - 2).cpu().numpy()
        # pyre-fixme[58]: `-` is not supported for operand types
        #  `Optional[torch.Tensor]` and `int`.
        logged_idx = (batch.tgt_out_idx - 2).cpu().numpy()
        score_bar = np.arange(self.slate_size, 0, -1)

        batch_dcg = []
        batch_ndcg = []
        batch_mean_ap = []
        batch_auc = []
        batch_base_dcg = []
        batch_base_ndcg = []
        batch_base_map = []
        batch_base_auc = []
        for i in range(batch_size):
            # no positive label in the slate or slate labels are all positive
            # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
            if (not torch.any(batch.position_reward[i].bool())) or (
                torch.all(batch.position_reward[i].bool())
            ):
                continue

            ranked_scores = np.zeros(self.slate_size)
            ranked_scores[ranked_idx[i]] = score_bar
            truth_scores = np.zeros(self.slate_size)
            truth_scores[logged_idx[i]] = batch.position_reward[i].cpu().numpy()
            base_scores = np.zeros(self.slate_size)
            base_scores[logged_idx[i]] = score_bar
            # average_precision_score accepts 1D arrays
            # dcg & ndcg accepts 2D arrays
            batch_mean_ap.append(average_precision_score(truth_scores, ranked_scores))
            batch_base_map.append(average_precision_score(truth_scores, base_scores))
            batch_auc.append(roc_auc_score(truth_scores, ranked_scores))
            batch_base_auc.append(roc_auc_score(truth_scores, base_scores))
            ranked_scores = np.expand_dims(ranked_scores, axis=0)
            truth_scores = np.expand_dims(truth_scores, axis=0)
            base_scores = np.expand_dims(base_scores, axis=0)
            batch_dcg.append(dcg_score(truth_scores, ranked_scores))
            batch_ndcg.append(ndcg_score(truth_scores, ranked_scores))
            batch_base_dcg.append(dcg_score(truth_scores, base_scores))
            batch_base_ndcg.append(ndcg_score(truth_scores, base_scores))

        self.reporter.log(
            eval_cross_entropy_loss=ce_loss,
            eval_dcg=torch.mean(torch.tensor(batch_dcg)).reshape(1),
            eval_ndcg=torch.mean(torch.tensor(batch_ndcg)).reshape(1),
            eval_mean_ap=torch.mean(torch.tensor(batch_mean_ap)).reshape(1),
            eval_auc=torch.mean(torch.tensor(batch_auc)).reshape(1),
            eval_base_dcg=torch.mean(torch.tensor(batch_base_dcg)).reshape(1),
            eval_base_ndcg=torch.mean(torch.tensor(batch_base_ndcg)).reshape(1),
            eval_base_map=torch.mean(torch.tensor(batch_base_map)).reshape(1),
            eval_base_auc=torch.mean(torch.tensor(batch_base_auc)).reshape(1),
        )
