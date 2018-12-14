#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Tuple

import torch
from ml.rl.evaluation.cpe import CpeEstimate
from ml.rl.evaluation.evaluation_data_page import EvaluationDataPage


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DoublyRobustEstimator:
    def estimate(
        self, edp: EvaluationDataPage
    ) -> Tuple[CpeEstimate, CpeEstimate, CpeEstimate]:
        # The score we would get if we evaluate the logged policy against itself
        denominator = float(torch.mean(edp.logged_rewards))

        if abs(denominator) < 1e-6:
            return (
                CpeEstimate(raw=0.0, normalized=0.0),
                CpeEstimate(raw=0.0, normalized=0.0),
                CpeEstimate(raw=0.0, normalized=0.0),
            )

        # For details, visit https://arxiv.org/pdf/1612.01205.pdf
        num_examples = edp.model_propensities.shape[0]

        if edp.model_rewards is None:
            # Fill with zero, equivalent to just doing IPS
            model_rewards = torch.zeros(edp.model_propensities.shape).float()
            direct_method_values = torch.zeros([num_examples, 1], dtype=torch.float32)
        else:
            model_rewards = edp.model_rewards
            direct_method_values = torch.sum(
                edp.model_propensities * model_rewards, dim=1, keepdim=True
            )

        target_propensity_for_action = torch.sum(
            edp.model_propensities * edp.action_mask, dim=1, keepdim=True
        )

        importance_weight = (
            target_propensity_for_action / edp.logged_propensities
        ).float()

        ips = importance_weight * edp.logged_rewards

        doubly_robust = (
            importance_weight
            * (edp.logged_rewards - edp.model_rewards_for_logged_action)
        ) + direct_method_values

        direct_method_score = float(torch.mean(direct_method_values))
        direct_method_estimate = CpeEstimate(
            raw=direct_method_score, normalized=direct_method_score / denominator
        )

        ips_score = float(torch.mean(ips))
        inverse_propensity_estimate = CpeEstimate(
            raw=ips_score, normalized=ips_score / denominator
        )

        dr_score = float(torch.mean(doubly_robust))
        doubly_robust_estimate = CpeEstimate(
            raw=dr_score, normalized=dr_score / denominator
        )

        return (
            direct_method_estimate,
            inverse_propensity_estimate,
            doubly_robust_estimate,
        )
