import math
from abc import ABC, abstractmethod
from typing import Optional, List

import torch
from reagent.mab.mab_algorithm import MABAlgo
from torch import Tensor


class BaseUCB(MABAlgo, ABC):
    """
    Base class for UCB-like Multi-Armed Bandits (MAB)

    Args:
        estimate_variance: If True, per-arm reward variance is estimated and we multiply thconfidence interval width
            by its square root
        alpha: Scalar multiplier for confidence interval width. Values above 1.0 make exploration more aggressive, below 1.0 less aggressive
    """

    def __init__(
        self,
        estimate_variance: bool = True,
        alpha: float = 1.0,
        *,
        n_arms: Optional[int] = None,
        arm_ids: Optional[List[str]] = None,
    ):
        super().__init__(n_arms=n_arms, arm_ids=arm_ids)
        self.estimate_variance = estimate_variance
        self.alpha = alpha

    @abstractmethod
    def get_ucb_scores(self) -> Tensor:
        pass

    def forward(self) -> Tensor:
        return self.get_ucb_scores()

    @property
    def var(self):
        # return empirical variance of rewards for each arm
        if self.estimate_variance:
            return self.total_sum_reward_squared_per_arm / self.total_n_obs_per_arm - (
                (self.total_sum_reward_per_arm / self.total_n_obs_per_arm) ** 2
            )
        else:
            return 1.0


class UCB1(BaseUCB):
    """
    Canonical implementation of UCB1
    Reference: https://www.cs.bham.ac.uk/internal/courses/robotics/lectures/ucb1.pdf
    """

    def get_ucb_scores(self) -> Tensor:
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
        ucb = avg_rewards + self.alpha * torch.sqrt(2 * log_t_over_ni * self.var)
        return torch.where(
            self.total_n_obs_per_arm > 0,
            ucb,
            torch.tensor(torch.inf, dtype=torch.float),
        )


class MetricUCB(BaseUCB):
    """
    This is an improvement over UCB1 which uses a more precise confidence radius, especially for small expected rewards.
    This algorithm has been constructed for Benroulli reward distributions.
    Reference: https://arxiv.org/pdf/0809.4882.pdf
    """

    def get_ucb_scores(self) -> Tensor:
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
        ucb = avg_rewards + self.alpha * (
            torch.sqrt(avg_rewards * log_t_over_ni) + log_t_over_ni
        )
        return torch.where(
            self.total_n_obs_per_arm > 0,
            ucb,
            torch.tensor(torch.inf, dtype=torch.float),
        )


def get_bernoulli_tuned_ucb_scores(n_obs_per_arm, num_success_per_arm):
    # a minimalistic function that implements Tuned UCB for Bernoulli bandit
    avg_rewards = n_obs_per_arm / num_success_per_arm
    log_t_over_ni = torch.log(torch.sum(n_obs_per_arm)) / num_success_per_arm
    per_arm_var_est = (
        avg_rewards
        - avg_rewards ** 2
        + torch.sqrt(
            2 * log_t_over_ni
        )  # additional term to make the estimate conservative (unlikely to underestimate)
    )
    return avg_rewards + torch.sqrt(log_t_over_ni * per_arm_var_est)
