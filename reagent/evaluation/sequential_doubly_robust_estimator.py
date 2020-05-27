#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List

import numpy as np
import torch
from reagent.evaluation.cpe import CpeEstimate, bootstrapped_std_error_of_mean
from reagent.evaluation.evaluation_data_page import EvaluationDataPage


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SequentialDoublyRobustEstimator:
    def __init__(self, gamma):
        self.gamma = gamma

    def estimate(self, edp: EvaluationDataPage) -> CpeEstimate:
        # For details, visit https://arxiv.org/pdf/1511.03722.pdf
        logged_rewards = edp.logged_rewards.squeeze()
        logged_propensities = edp.logged_propensities.squeeze()

        num_examples = edp.logged_rewards.shape[0]

        estimated_state_values = torch.sum(
            edp.model_propensities * edp.model_values, dim=1
        )

        estimated_q_values_for_logged_action = torch.sum(
            edp.model_values * edp.action_mask, dim=1
        )

        target_propensity_for_action = torch.sum(
            edp.model_propensities * edp.action_mask, dim=1
        )

        assert target_propensity_for_action.shape == logged_propensities.shape, (
            "Invalid shape: "
            + str(target_propensity_for_action.shape)
            + " != "
            + str(logged_propensities.shape)
        )
        assert (
            target_propensity_for_action.shape
            == estimated_q_values_for_logged_action.shape
        ), (
            "Invalid shape: "
            + str(target_propensity_for_action.shape)
            + " != "
            + str(estimated_q_values_for_logged_action.shape)
        )
        assert target_propensity_for_action.shape == logged_rewards.shape, (
            "Invalid shape: "
            + str(target_propensity_for_action.shape)
            + " != "
            + str(logged_rewards.shape)
        )
        importance_weight = target_propensity_for_action / logged_propensities

        doubly_robusts: List[float] = []
        episode_values: List[float] = []

        assert edp.mdp_id is not None
        i = 0
        last_episode_end = -1
        while i < num_examples:
            # calculate the doubly-robust Q-value for one episode
            if i == num_examples - 1 or edp.mdp_id[i] != edp.mdp_id[i + 1]:
                episode_end = i
                episode_value = 0.0
                doubly_robust = 0.0
                for j in range(episode_end, last_episode_end, -1):
                    doubly_robust = estimated_state_values[j] + importance_weight[j] * (
                        logged_rewards[j]
                        + self.gamma * doubly_robust
                        - estimated_q_values_for_logged_action[j]
                    )
                    episode_value *= self.gamma
                    episode_value += logged_rewards[j]
                if episode_value > 1e-6 or episode_value < -1e-6:
                    doubly_robusts.append(float(doubly_robust))
                    episode_values.append(float(episode_value))
                last_episode_end = episode_end
            i += 1

        assert len(doubly_robusts) > 0, (
            f"No valid doubly robusts data is generated. "
            f"Logged_rewards={logged_rewards}, importance_weight={importance_weight},"
            f" estimated_q_values_for_logged_action={estimated_q_values_for_logged_action}"
            f" estimated_state_values={estimated_state_values}, gamma={self.gamma}"
            f" Did you specify wrong metric names?"
        )

        # pyre-fixme[9]: doubly_robusts has type `List[float]`; used as `ndarray`.
        doubly_robusts = np.array(doubly_robusts)
        dr_score = float(np.mean(doubly_robusts))
        dr_score_std_error = bootstrapped_std_error_of_mean(doubly_robusts)

        # pyre-fixme[9]: episode_values has type `List[float]`; used as `ndarray`.
        episode_values = np.array(episode_values)
        logged_policy_score = np.mean(episode_values)
        if logged_policy_score < 1e-6:
            logger.warning(
                "Can't normalize SDR-CPE because of small or negative logged_policy_score"
            )
            return CpeEstimate(
                raw=dr_score,
                normalized=0.0,
                raw_std_error=dr_score_std_error,
                normalized_std_error=0.0,
            )
        return CpeEstimate(
            raw=dr_score,
            normalized=dr_score / logged_policy_score,
            raw_std_error=dr_score_std_error,
            normalized_std_error=dr_score_std_error / logged_policy_score,
        )
