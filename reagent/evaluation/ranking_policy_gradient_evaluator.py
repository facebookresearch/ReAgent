#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import Optional

import numpy as np

# @manual=third-party//scipy:scipy-py
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from reagent.evaluation.doubly_robust_estimator import DoublyRobustEstimator
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
        self.advantages = []
        self.logged_slate_log_probs = []
        self.ranked_slate_probs = []
        self.baseline_loss = []
        self.calc_cpe = calc_cpe
        self.reward_network = reward_network
        # Evaluate greedy/non-greedy version of the ranking model
        self.eval_data_pages_g: Optional[EvaluationDataPage] = None
        self.eval_data_pages_ng: Optional[EvaluationDataPage] = None

    @torch.no_grad()
    def evaluate(self, eval_tdp: PreprocessedTrainingBatch) -> None:
        seq2slate_net = self.trainer.seq2slate_net
        seq2slate_net_prev_mode = seq2slate_net.training
        seq2slate_net.eval()

        logged_slate_log_prob = (
            seq2slate_net(
                eval_tdp.training_input, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE
            )
            .log_probs.detach()
            .flatten()
            .cpu()
            .numpy()
        )

        if self.trainer.baseline_net:
            baseline_net = self.trainer.baseline_net
            # pyre-fixme[16]: `Optional` has no attribute `training`.
            baseline_net_prev_mode = baseline_net.training
            # pyre-fixme[16]: `Optional` has no attribute `eval`.
            baseline_net.eval()
            # pyre-fixme[29]: `Optional[reagent.models.seq2slate.BaselineNet]` is
            #  not a function.
            b = baseline_net(eval_tdp.training_input).detach()
            self.baseline_loss.append(
                F.mse_loss(b, eval_tdp.training_input.slate_reward).item()
            )
            # pyre-fixme[16]: `Optional` has no attribute `train`.
            baseline_net.train(baseline_net_prev_mode)
        else:
            b = torch.zeros_like(eval_tdp.training_input.slate_reward)
            self.baseline_loss.append(0.0)

        # pyre-fixme[16]: `Optional` has no attribute `__sub__`.
        advantage = (eval_tdp.training_input.slate_reward - b).flatten().cpu().numpy()
        self.advantages.append(advantage)
        self.logged_slate_log_probs.append(logged_slate_log_prob)

        ranked_slate_output = seq2slate_net(
            eval_tdp.training_input, Seq2SlateMode.RANK_MODE, greedy=True
        )
        ranked_slate_prob = (
            torch.prod(
                torch.gather(
                    ranked_slate_output.ranked_tgt_out_probs,
                    2,
                    ranked_slate_output.ranked_tgt_out_idx.unsqueeze(-1),
                ).squeeze(),
                -1,
            )
            .cpu()
            .numpy()
        )
        self.ranked_slate_probs.append(ranked_slate_prob)

        seq2slate_net.train(seq2slate_net_prev_mode)

        if not self.calc_cpe:
            return

        edp_g = EvaluationDataPage.create_from_tensors_seq2slate(
            seq2slate_net,
            # pyre-fixme[6]: Expected `Module` for 2nd param but got
            #  `Optional[nn.Module]`.
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
            # pyre-fixme[6]: Expected `Module` for 2nd param but got
            #  `Optional[nn.Module]`.
            self.reward_network,
            eval_tdp.training_input,
            eval_greedy=False,
        )
        if self.eval_data_pages_ng is None:
            self.eval_data_pages_ng = edp_ng
        else:
            self.eval_data_pages_ng = self.eval_data_pages_ng.append(edp_ng)

    @torch.no_grad()
    def evaluate_post_training(self):
        # One indicator of successful training is that sequences with large
        # advantages should have large log probabilities
        kendall_tau = stats.kendalltau(
            np.hstack(self.logged_slate_log_probs), np.hstack(self.advantages)
        )
        logger.info(
            f"kendall_tau={kendall_tau.correlation}, p-value={kendall_tau.pvalue}"
        )
        eval_res = {
            "kendall_tau": kendall_tau.correlation,
            "kendall_tau_p_value": kendall_tau.pvalue,
            "baseline": np.mean(self.baseline_loss),
        }

        if self.calc_cpe:
            eval_res["mean_ips_greedy"] = self._mean_ips(
                self.eval_data_pages_g, greedy=True
            )
            eval_res["mean_ips_non_greedy"] = self._mean_ips(
                self.eval_data_pages_ng, greedy=False
            )

            doubly_robust_estimator = DoublyRobustEstimator()
            (
                direct_method,
                inverse_propensity,
                doubly_robust,
            ) = doubly_robust_estimator.estimate(self.eval_data_pages_g)
            eval_res["cpe_dm_raw"] = direct_method.raw
            eval_res["cpe_dm_normalized"] = direct_method.normalized
            eval_res["cpe_ips_raw_greedy"] = inverse_propensity.raw
            eval_res["cpe_ips_normalized_greedy"] = inverse_propensity.normalized
            eval_res["cpe_dr_raw"] = doubly_robust.raw
            eval_res["cpe_dr_normalized"] = doubly_robust.normalized

            _, inverse_propensity, _ = doubly_robust_estimator.estimate(
                self.eval_data_pages_ng
            )
            eval_res["cpe_ips_raw_non_greedy"] = inverse_propensity.raw
            eval_res["cpe_ips_normalized_non_greedy"] = inverse_propensity.normalized

            eval_res["ranked_slate_probs"] = np.mean(self.ranked_slate_probs)

        self.advantages = []
        self.logged_slate_log_probs = []
        self.ranked_slate_probs = []
        self.baseline_loss = []
        self.eval_data_pages_g = None
        self.eval_data_pages_ng = None

        return eval_res

    def _mean_ips(self, eval_data_page: EvaluationDataPage, greedy: bool):
        assert (
            (
                eval_data_page.action_mask.shape
                == eval_data_page.logged_propensities.shape
                == eval_data_page.model_propensities.shape
            )
            and len(eval_data_page.logged_propensities.shape) == 2
            and eval_data_page.logged_propensities.shape[1] == 1
        ), (
            f"{eval_data_page.action_mask.shape} "
            f"{eval_data_page.model_propensities.shape} "
            f"{eval_data_page.logged_propensities.shape}"
        )
        if greedy:
            return np.mean(
                eval_data_page.action_mask.cpu().numpy()
                / eval_data_page.logged_propensities.cpu().numpy()
            )
        return np.mean(
            eval_data_page.model_propensities.cpu().numpy()
            / eval_data_page.logged_propensities.cpu().numpy()
        )
