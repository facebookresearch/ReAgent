#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import math
import time
from abc import abstractmethod
from typing import Dict, Optional

import torch
from reagent.core.types import CBInput


logger = logging.getLogger(__name__)


class MABBaseModel(torch.nn.Module):
    """
    A base class for Multi-Armed Bandits (MAB). These are non-contextual bandit models.
    IMPORTANT USAGE GUIDILINES:
    1. Use manual optimization as there is no loss function to optimize.
    2. Distributed training is not supported yet, so only run this with a single process.
        Distributed training could be added if needed using `torch.distributed.all_gather_object()`
    3. A minimal implementation of a specific MAB algorithm needs to implement only `get_score_single_arm_id()` method
    4. The base class keeps track of: (1) number of observations per arm (2) sum of rewards per arm (3) sum of squared rewards per arm.
        If a specific MAB algorithm wants to keep track of other metrics, it needs to override the `learn()` method.
    5. This model is intended to be used only for Offline Evaluation (e.g. hyperparameter search, algorithm comparison).
        DO NOT TRY TO SERVE THIS MODEL IN PRODUCTION.

    Args:
        ucb_alpha: The coefficient on the standard deviation in UCB formula.
            Set it to 0 to predict the expected value instead of UCB.
        min_explore_steps: Minimum number of observations (per arm) necessary before we
            start returning non-default UCB scores.
        default_score: The default score returned instead of UCB score during
            first `min_explore_steps` for each arm. The detault is `inf`, which should prioritize
            this arm ahead of any arms that already have more than `min_explore_steps` observations.
        estimate_variance: Whether to estimate reward variance using sample variance or not. If not,
            default value of 1.0 is used
        min_variance: If estimating variance, this value is applied to the estimate as the minimum.

    Outputs:
    Dict {"pred_label": pred_label, "pred_sigma": pred_sigma, "ucb": ucb}
    """

    def __init__(
        self,
        ucb_alpha: float = 1.0,
        min_explore_steps: int = 1,
        default_score: float = float("inf"),
        estimate_variance: bool = True,
        min_variance: float = 0.0,
        input_dim: int = 1,  # ununsed argument, keeping it for compatibility with parent class
    ):
        super().__init__()
        self.num_obs_total = 0
        self.obs_count_by_arm_id = {}
        self.sum_reward_by_arm_id = {}
        self.sum_squared_reward_by_arm_id = {}
        self.ucb_alpha = ucb_alpha
        self.min_explore_steps = min_explore_steps
        self.default_score = default_score
        self.estimate_variance = estimate_variance
        self.min_variance = min_variance

        # add a dummy parameter so that DDP doesn't compain about lack of parameters with gradient
        self.dummy_param = torch.nn.parameter.Parameter(torch.zeros(1))

    def learn(self, batch: CBInput) -> None:
        """
        Process a batch of data and update the per-arm counters.
        """
        # use time module to check execution time
        start = time.time()
        assert batch.arms is not None
        assert batch.reward is not None
        assert batch.action is not None
        # inefficient for loop is used for V0. Might improve performance later if necessary.
        weights = batch.effective_weight
        for arm_ids_one_obs, reward, weight, action in zip(
            batch.arms.tolist(),
            batch.reward.squeeze().tolist(),
            weights.squeeze().tolist(),
            batch.action.squeeze().tolist(),
        ):
            arm_id = arm_ids_one_obs[action]
            self.num_obs_total += weight
            self.obs_count_by_arm_id[arm_id] = (
                self.obs_count_by_arm_id.get(arm_id, 0.0) + weight
            )
            self.sum_reward_by_arm_id[arm_id] = (
                self.sum_reward_by_arm_id.get(arm_id, 0.0) + weight * reward
            )
            self.sum_squared_reward_by_arm_id[arm_id] = (
                self.sum_squared_reward_by_arm_id.get(arm_id, 0.0) + weight * reward**2
            )
        end = time.time()
        logger.info(f"Batch learn time: {end - start}")

    def get_variance(self, arm_id: int) -> float:
        """
        Return empirical variance of rewards for the arm. Minimum variance value is applied to deal
            with cases where empirical rewards are all 0.
        If not estimating variance, return 1.0.
        """
        if self.estimate_variance:
            sample_variance = (
                self.sum_squared_reward_by_arm_id[arm_id]
                / self.obs_count_by_arm_id[arm_id]
            ) - (
                (self.sum_reward_by_arm_id[arm_id] / self.obs_count_by_arm_id[arm_id])
                ** 2
            )
            return max(
                self.min_variance,
                sample_variance,
            )
        else:
            return 1.0

    def get_score_single_arm_id_if_enough_data(
        self, arm_id: int, ucb_alpha: float
    ) -> Dict[str, float]:
        """
        Check if we have enough data for this arm.
            - If yes, calculate the UCB score for this arm
            - If no, return the default UCB score
        """
        if self.obs_count_by_arm_id.get(arm_id, 0) >= self.min_explore_steps:
            return self.get_score_single_arm_id(arm_id, ucb_alpha)
        else:
            return {
                "pred_label": 0.0,
                "pred_sigma": 0.0,
                "ucb": self.default_score,
            }

    @abstractmethod
    def get_score_single_arm_id(
        self, arm_id: int, ucb_alpha: float
    ) -> Dict[str, float]:
        """
        Get the scores (predicted mu, sigma and UCB) for a single arm_id.
        """
        pass

    def forward(
        self, arm_ids: torch.Tensor, ucb_alpha: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get the scores (predicted mu, sigma and UCB) for a batch of arm_ids.
        """
        start = time.time()
        if ucb_alpha is None:
            ucb_alpha = self.ucb_alpha
        all_scores = [
            [
                self.get_score_single_arm_id_if_enough_data(arm_id, ucb_alpha)
                for arm_id in single_obs_arm_ids.tolist()
            ]
            for single_obs_arm_ids in arm_ids
        ]
        check1 = time.time()
        logger.info(f"Check1 forward time: {check1 - start}")
        ret = {
            k: torch.tensor(
                [
                    [single_arm_scores[k] for single_arm_scores in slate_score]
                    for slate_score in all_scores
                ]
            )
            for k in ["pred_label", "pred_sigma", "ucb"]
        }
        end = time.time()
        logger.info(f"Batch forward time: {end - start}")
        # from reagent.core.utils import ForkedPdb
        # ForkedPdb().set_trace()
        return ret


class UCB1MAB(MABBaseModel):
    """
    Canonical implementation of UCB1
    Reference: https://www.cs.bham.ac.uk/internal/courses/robotics/lectures/ucb1.pdf
    """

    def get_score_single_arm_id(
        self, arm_id: int, ucb_alpha: float
    ) -> Dict[str, float]:
        mu = self.sum_reward_by_arm_id[arm_id] / self.obs_count_by_arm_id[arm_id]
        sigma = self.get_variance(arm_id) ** 0.5
        return {
            "pred_label": mu,
            "pred_sigma": sigma,
            "ucb": mu
            + ucb_alpha
            * sigma
            * math.sqrt(
                2.0
                * math.log(self.num_obs_total + 1.0)
                / self.obs_count_by_arm_id[arm_id]
            ),
        }
