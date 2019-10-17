#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

import numpy as np

# @manual=third-party//scipy:scipy-py
import scipy.stats as stats
import torch
from ml.rl.models.seq2slate import LOG_PROB_MODE
from ml.rl.training.ranking.seq2slate_trainer import Seq2SlateTrainer
from ml.rl.types import PreprocessedTrainingBatch


logger = logging.getLogger(__name__)


class RankingEvaluator:
    """ Evaluate ranking models """

    def __init__(self, trainer: Seq2SlateTrainer) -> None:
        self.trainer = trainer
        self.advantages = []
        self.log_probs = []

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
        advantage = (
            (
                eval_tdp.training_input.slate_reward
                - baseline_net(eval_tdp.training_input).squeeze().detach()
            )
            .flatten()
            .cpu()
            .numpy()
        )
        self.advantages.append(advantage)
        self.log_probs.append(log_prob)

        seq2slate_net.train(seq2slate_net_prev_mode)
        baseline_net.train(baseline_net_prev_mode)

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
        }

        self.advantages = []
        self.log_probs = []

        return eval_res
