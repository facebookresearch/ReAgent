#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from reagent.core.tracker import observable
from reagent.model_utils.seq2slate_utils import Seq2SlateMode
from reagent.types import PreprocessedTrainingBatch
from sklearn.metrics import (
    average_precision_score,
    dcg_score,
    ndcg_score,
    roc_auc_score,
)


logger = logging.getLogger(__name__)


@dataclass
class ListwiseRankingMetrics:
    ndcg: Optional[float] = 0.0
    dcg: Optional[float] = 0.0
    mean_ap: Optional[float] = 0.0
    cross_entropy_loss: Optional[float] = 0.0


@observable(
    cross_entropy_loss=torch.Tensor,
    dcg=torch.Tensor,
    ndcg=torch.Tensor,
    mean_ap=torch.Tensor,
    auc=torch.Tensor,
    base_dcg=torch.Tensor,
    base_ndcg=torch.Tensor,
    base_map=torch.Tensor,
    base_auc=torch.Tensor,
)
class RankingListwiseEvaluator:
    """ Evaluate listwise ranking models on common ranking metrics """

    def __init__(self, seq2slate_net, slate_size: int, calc_cpe: bool) -> None:
        self.seq2slate_net = seq2slate_net
        self.slate_size = slate_size
        self.calc_cpe = calc_cpe
        self.ndcg = []
        self.dcg = []
        self.mean_ap = []
        self.base_dcg = []
        self.base_ndcg = []
        self.base_map = []
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    # pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
    #  its type `no_grad` is not callable.
    @torch.no_grad()
    def evaluate(self, eval_tdp: PreprocessedTrainingBatch) -> None:
        seq2slate_net_prev_mode = self.seq2slate_net.training
        self.seq2slate_net.eval()

        eval_input = eval_tdp.training_input
        # pyre-fixme[16]: `Optional` has no attribute `shape`.
        batch_size = eval_input.position_reward.shape[0]

        # shape: batch_size, tgt_seq_len
        encoder_scores = self.seq2slate_net(
            eval_input, mode=Seq2SlateMode.ENCODER_SCORE_MODE
        ).encoder_scores
        assert (
            encoder_scores.shape[1]
            == eval_input.position_reward.shape[1]
            == self.slate_size
        )
        ce_loss = self.kl_loss(
            self.log_softmax(encoder_scores), eval_input.position_reward
        ).item()

        self.seq2slate_net.train(seq2slate_net_prev_mode)

        if not self.calc_cpe:
            # pyre-fixme[16]: `RankingListwiseEvaluator` has no attribute
            #  `notify_observers`.
            self.notify_observers(cross_entropy_loss=ce_loss)
            return

        # shape: batch_size, tgt_seq_len
        ranking_output = self.seq2slate_net(eval_input, mode=Seq2SlateMode.RANK_MODE)
        # pyre-fixme[16]: `int` has no attribute `cpu`.
        ranked_idx = (ranking_output.ranked_tgt_out_idx - 2).cpu().numpy()
        # pyre-fixme[6]: Expected `int` for 1st param but got `Optional[torch.Tensor]`.
        logged_idx = (eval_input.tgt_out_idx - 2).cpu().numpy()
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
            if (not torch.any(eval_input.position_reward[i].bool())) or (
                torch.all(eval_input.position_reward[i].bool())
            ):
                continue

            ranked_scores = np.zeros(self.slate_size)
            ranked_scores[ranked_idx[i]] = score_bar
            truth_scores = np.zeros(self.slate_size)
            truth_scores[logged_idx[i]] = eval_input.position_reward[i].cpu().numpy()
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

        self.notify_observers(
            cross_entropy_loss=ce_loss,
            dcg=torch.mean(torch.tensor(batch_dcg)).reshape(1),
            ndcg=torch.mean(torch.tensor(batch_ndcg)).reshape(1),
            mean_ap=torch.mean(torch.tensor(batch_mean_ap)).reshape(1),
            auc=torch.mean(torch.tensor(batch_auc)).reshape(1),
            base_dcg=torch.mean(torch.tensor(batch_base_dcg)).reshape(1),
            base_ndcg=torch.mean(torch.tensor(batch_base_ndcg)).reshape(1),
            base_map=torch.mean(torch.tensor(batch_base_map)).reshape(1),
            base_auc=torch.mean(torch.tensor(batch_base_auc)).reshape(1),
        )

    @torch.no_grad()
    def evaluate_post_training(self):
        pass
