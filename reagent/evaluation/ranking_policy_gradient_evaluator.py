#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

# @manual=third-party//scipy:scipy-py
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from reagent.evaluation.evaluation_data_page import EvaluationDataPage
from reagent.models.seq2slate import Seq2SlateMode
from reagent.training.ranking.seq2slate_trainer import Seq2SlateTrainer
from reagent.types import PreprocessedTrainingBatch


logger = logging.getLogger(__name__)


class RankingPolicyGradientEvaluator:
    """ Evaluate ranking models that are learned through policy gradient """

    def __init__(
        self,
        trainer: Seq2SlateTrainer,
        calc_cpe: bool,
        reward_network: Optional[nn.Module] = None,
    ) -> None:
        assert not calc_cpe or reward_network is not None
        self.trainer = trainer
        self.calc_cpe = calc_cpe
        self.reward_network = reward_network
        self.reporter = None

        # Evaluate greedy/non-greedy version of the ranking model
        self.eval_data_pages_g: Optional[EvaluationDataPage] = None
        self.eval_data_pages_ng: Optional[EvaluationDataPage] = None

    @torch.no_grad()
    def evaluate(self, eval_tdp: PreprocessedTrainingBatch) -> None:
        seq2slate_net = self.trainer.seq2slate_net
        seq2slate_net_prev_mode = seq2slate_net.training
        seq2slate_net.eval()

        logged_slate_rank_prob = torch.exp(
            seq2slate_net(
                eval_tdp.training_input, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE
            )
            .log_probs.detach()
            .flatten()
            .cpu()
        )

        eval_baseline_loss = torch.tensor([0.0]).reshape(1)
        if self.trainer.baseline_net:
            baseline_net = self.trainer.baseline_net
            # pyre-fixme[16]: `Optional` has no attribute `training`.
            baseline_net_prev_mode = baseline_net.training
            # pyre-fixme[16]: `Optional` has no attribute `eval`.
            baseline_net.eval()
            # pyre-fixme[29]: `Optional[reagent.models.seq2slate.BaselineNet]` is
            #  not a function.
            b = baseline_net(eval_tdp.training_input).detach()
            eval_baseline_loss = (
                F.mse_loss(b, eval_tdp.training_input.slate_reward).cpu().reshape(1)
            )
            # pyre-fixme[16]: `Optional` has no attribute `train`.
            baseline_net.train(baseline_net_prev_mode)
        else:
            b = torch.zeros_like(eval_tdp.training_input.slate_reward)

        eval_advantage = (
            # pyre-fixme[6]: `-` is not supported for operand types
            #  `Optional[torch.Tensor]` and `Any`.
            (eval_tdp.training_input.slate_reward - b)
            .flatten()
            .cpu()
        )

        ranked_slate_output = seq2slate_net(
            eval_tdp.training_input, Seq2SlateMode.RANK_MODE, greedy=True
        )
        ranked_slate_rank_prob = torch.prod(
            torch.gather(
                ranked_slate_output.ranked_tgt_out_probs,
                2,
                ranked_slate_output.ranked_tgt_out_idx.unsqueeze(-1),
            ).squeeze(),
            -1,
        ).cpu()

        seq2slate_net.train(seq2slate_net_prev_mode)

        if not self.calc_cpe:
            return

        edp_g = EvaluationDataPage.create_from_tensors_seq2slate(
            seq2slate_net,
            self.reward_network,
            eval_tdp.training_input,
            eval_greedy=True,
        )
        if self.eval_data_pages_g is None:
            self.eval_data_pages_g = edp_g
        else:
            # pyre-fixme[16]: `Optional` has no attribute `append`.
            self.eval_data_pages_g = self.eval_data_pages_g.append(edp_g)

        edp_ng = EvaluationDataPage.create_from_tensors_seq2slate(
            seq2slate_net,
            self.reward_network,
            eval_tdp.training_input,
            eval_greedy=False,
        )
        if self.eval_data_pages_ng is None:
            self.eval_data_pages_ng = edp_ng
        else:
            self.eval_data_pages_ng = self.eval_data_pages_ng.append(edp_ng)

        self.reporter.report_evaluation_minibatch(
            eval_baseline_loss=eval_baseline_loss,
            eval_advantages=eval_advantage,
            logged_slate_rank_probs=logged_slate_rank_prob,
            ranked_slate_rank_probs=ranked_slate_rank_prob,
        )

    @torch.no_grad()
    def finish(self):
        self.reporter.report_evaluation_epoch(
            eval_data_pages_g=self.eval_data_pages_g,
            eval_data_pages_ng=self.eval_data_pages_ng,
        )
        self.eval_data_pages_g = None
        self.eval_data_pages_ng = None

    def evaluate_one_shot(self, edp: EvaluationDataPage):
        pass
