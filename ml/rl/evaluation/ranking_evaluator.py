#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import Optional

import numpy as np

# @manual=third-party//scipy:scipy-py
import scipy.stats as stats
import torch
import torch.nn.functional as F
from ml.rl.evaluation.evaluation_data_page import EvaluationDataPage
from ml.rl.models.seq2slate import LOG_PROB_MODE
from ml.rl.training.ranking.seq2slate_trainer import Seq2SlateTrainer
from ml.rl.types import PreprocessedTrainingBatch


logger = logging.getLogger(__name__)


class RankingEvaluator:
    """ Evaluate ranking models """

    def __init__(self, trainer: Seq2SlateTrainer, calc_cpe: bool) -> None:
        self.trainer = trainer
        self.advantages = []
        self.log_probs = []
        self.baseline_loss = []
        self.calc_cpe = calc_cpe
        self.eval_data_pages: Optional[EvaluationDataPage] = None

    @torch.no_grad()
    def evaluate(self, eval_tdp: PreprocessedTrainingBatch):
        seq2slate_net = self.trainer.seq2slate_net
        baseline_net = self.trainer.baseline_net

        seq2slate_net_prev_mode = seq2slate_net.training
        baseline_net_prev_mode = baseline_net.training
        seq2slate_net.eval()
        baseline_net.eval()

        log_prob = (
            seq2slate_net(eval_tdp.training_input, mode=LOG_PROB_MODE)
            .log_probs.detach()
            .flatten()
            .cpu()
            .numpy()
        )
        b = baseline_net(eval_tdp.training_input).squeeze().detach()
        advantage = (eval_tdp.training_input.slate_reward - b).flatten().cpu().numpy()

        self.baseline_loss.append(
            F.mse_loss(b, eval_tdp.training_input.slate_reward).item()
        )
        self.advantages.append(advantage)
        self.log_probs.append(log_prob)

        seq2slate_net.train(seq2slate_net_prev_mode)
        baseline_net.train(baseline_net_prev_mode)

        if not self.calc_cpe:
            return

        # TODO: create evaluation data page
        # edp = EvaluationDataPage.create_from_training_batch(tdp, self.trainer)
        # if self.eval_data_pages = None:
        #     self.eval_data_pages = edp
        # else:
        #     self.eval_data_pages = self.eval_data_pages.append(edp)

    @torch.no_grad()
    def evaluate_post_training(self):
        # One indicator of successful training is that sequences with large
        # advantages should have large log probabilities
        kendall_tau = stats.kendalltau(
            np.hstack(self.log_probs), np.hstack(self.advantages)
        )
        logger.info(
            f"kendall_tau={kendall_tau.correlation}, p-value={kendall_tau.pvalue}"
        )
        eval_res = {
            "kendall_tau": kendall_tau.correlation,
            "kendall_tau_p_value": kendall_tau.pvalue,
            "baseline": np.mean(self.baseline_loss),
        }

        # TODO: add doubly robust estimator
        # if self.calc_cpe:
        # doubly_robust_estimator = DoublyRobustEstimator()
        # direct_method, inverse_propensity, doubly_robust = doubly_robust_estimator.estimate(
        #     self.eval_data_pages
        # )

        self.advantages = []
        self.log_probs = []
        self.baseline_loss = []
        self.eval_data_pages = None

        return eval_res
