#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import math
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from reagent.mab.mab_algorithm import MABAlgo
from torch import Tensor


class BaseUCB(MABAlgo, ABC):
    """
    Base class for UCB-like Multi-Armed Bandits (MAB)

    Args:
        estimate_variance: If True, per-arm reward variance is estimated and we multiply thconfidence interval width
            by its square root
        min_variance: The lower bound applied to the estimated variance. If variance is not estimated, this value is used instead of an estimate.
        alpha: Scalar multiplier for confidence interval width. Values above 1.0 make exploration more aggressive, below 1.0 less aggressive
    """

    def __init__(
        self,
        randomize_ties: bool = True,
        estimate_variance: bool = True,
        min_variance: float = 0.0,
        alpha: float = 1.0,
        min_num_obs_per_arm: int = 1,
        *,
        n_arms: Optional[int] = None,
        arm_ids: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            arm_ids=arm_ids,
            randomize_ties=randomize_ties,
            min_num_obs_per_arm=min_num_obs_per_arm,
        )
        self.estimate_variance = estimate_variance
        self.min_variance = torch.tensor(min_variance)
        self.alpha = alpha

    @property
    def var(self):
        # return empirical variance of rewards for each arm
        if self.estimate_variance:
            return torch.fmax(
                self.min_variance,
                self.total_sum_reward_squared_per_arm / self.total_n_obs_per_arm
                - ((self.total_sum_reward_per_arm / self.total_n_obs_per_arm) ** 2),
            )
        else:
            return self.min_variance


class UCB1(BaseUCB):
    """
    Canonical implementation of UCB1
    Reference: https://www.cs.bham.ac.uk/internal/courses/robotics/lectures/ucb1.pdf
    """

    def get_scores(self) -> Tensor:
        """
        Get per-arm UCB scores. The formula is
        UCB_i = AVG([rewards_i]) + SQRT(2*LN(T)/N_i*VAR)
        VAR=1 if estimate_variance==False, otherwise VAR=AVG([rewards_i**2]) - AVG([rewards_i])**2

        Returns:
            Tensor: An array of UCB scores (one per arm)
        """
        avg_rewards = self.get_avg_reward_values()
        log_t_over_ni = (
            math.log(self.total_n_obs_all_arms + 1) / self.total_n_obs_per_arm
        )
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `float`.
        return avg_rewards + self.alpha * torch.sqrt(2 * log_t_over_ni * self.var)


class MetricUCB(BaseUCB):
    """
    This is an improvement over UCB1 which uses a more precise confidence radius, especially for small expected rewards.
    This algorithm has been constructed for Bernoulli reward distributions.
    Reference: https://arxiv.org/pdf/0809.4882.pdf
    """

    def get_scores(self) -> Tensor:
        """
        Get per-arm UCB scores. The formula is
        UCB_i = AVG([rewards_i]) + SQRT(AVG([rewards_i]) * LN(T+1)/N_i) + LN(T+1)/N_i

        Returns:
            Tensor: An array of UCB scores (one per arm)
        """
        avg_rewards = self.get_avg_reward_values()
        log_t_over_ni = (
            math.log(self.total_n_obs_all_arms + 1) / self.total_n_obs_per_arm
        )
        return avg_rewards + self.alpha * (
            torch.sqrt(avg_rewards * log_t_over_ni) + log_t_over_ni
        )


class UCBTuned(BaseUCB):
    """
    Implementation of the UCB-Tuned algorithm from Section 4 of  https://link.springer.com/content/pdf/10.1023/A:1013689704352.pdf
    Biggest difference from basic UCB is that per-arm reward variance is estimated.
    IMPORTANT: This algorithm should only be used if the rewards of each arm have Bernoulli distribution.
    """

    def get_scores(self) -> Tensor:
        """
        Get per-arm UCB scores. The formula is
        UCB_i = AVG([rewards_i]) + SQRT(LN(T)/N_i * min(V_i, 0.25))
        where V_i is a conservative variance estimate of arm i:
            V_i = AVG([rewards_i**2]) - AVG([rewards_i])**2 + sqrt(2ln(t) / n_i)

        Returns:
            Tensor: An array of UCB scores (one per arm)
        """
        avg_rewards = self.get_avg_reward_values()
        log_t_over_ni = (
            math.log(self.total_n_obs_all_arms + 1) / self.total_n_obs_per_arm
        )
        per_arm_var_est = (
            self.total_sum_reward_squared_per_arm / self.total_n_obs_per_arm
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            - avg_rewards**2
            + torch.sqrt(
                # pyre-fixme[6]: For 1st param expected `Tensor` but got `float`.
                2 * log_t_over_ni
            )  # additional term to make the estimate conservative (unlikely to underestimate)
        )
        return avg_rewards + self.alpha * torch.sqrt(
            log_t_over_ni * torch.fmin(per_arm_var_est, torch.tensor(0.25))
        )


def get_bernoulli_ucb_tuned_scores(
    n_obs_per_arm: Tensor, num_success_per_arm: Tensor
) -> Tensor:
    """
    a minimalistic function that implements UCB-Tuned for Bernoulli bandit
    it's here only to benchmark execution time penalty incurred by the class-based implementation
    """
    avg_rewards = num_success_per_arm / n_obs_per_arm
    log_t_over_ni = torch.log(torch.sum(n_obs_per_arm)) / n_obs_per_arm
    per_arm_var_est = (
        avg_rewards
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        - avg_rewards**2
        + torch.sqrt(
            2 * log_t_over_ni
        )  # additional term to make the estimate conservative (unlikely to underestimate)
    )
    # pyre-fixme[6]: For 2nd param expected `Tensor` but got `float`.
    return avg_rewards + torch.sqrt(log_t_over_ni * torch.fmin(per_arm_var_est, 0.25))
