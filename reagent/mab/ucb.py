import math
from abc import ABC, abstractmethod
from typing import Union, Optional, List

import torch
from reagent.mab.mab_algorithm import MABAlgo
from torch import Tensor


class BaseUCB(MABAlgo, ABC):
    """
    Base class for UCB-like Multi-Armed Bandits (MAB)
    """

    @abstractmethod
    def get_ucb_scores(self):
        pass

    def __repr__(self):
        t = ", ".join(
            f"{v:.3f} ({int(n)})"
            for v, n in zip(self.get_avg_reward_values(), self.total_n_obs_per_arm)
        )
        return f"UCB({self.n_arms} arms; {t}"

    def forward(self):
        return self.get_ucb_scores()


class UCB1(BaseUCB):
    """
    Canonical implementation of UCB1
    Reference: https://www.cs.bham.ac.uk/internal/courses/robotics/lectures/ucb1.pdf
    """

    def get_ucb_scores(self):
        """
        Get per-arm UCB scores. The formula is
        UCB_i = AVG([rewards_i]) + SQRT(2*LN(T)/N_i)

        Returns:
            Tensor: An array of UCB scores (one per arm)
        """
        avg_rewards = self.get_avg_reward_values()
        log_t_over_ni = (
            math.log(self.total_n_obs_all_arms + 1) / self.total_n_obs_per_arm
        )
        ucb = avg_rewards + torch.sqrt(2 * log_t_over_ni)
        return torch.where(
            self.total_n_obs_per_arm > 0,
            ucb,
            torch.tensor(torch.inf, dtype=torch.float),
        )


class UCBTuned(BaseUCB):
    """
    Implementation of the UCB-Tuned algorithm from Section 4 of  https://link.springer.com/content/pdf/10.1023/A:1013689704352.pdf
    Biggest difference from basic UCB is that per-arm reward variance is estimated.
    """

    def get_ucb_scores(self) -> Tensor:
        """
        Get per-arm UCB scores. The formula is
        UCB_i = AVG([rewards_i]) + SQRT(LN(T)/N_i * V_i)
        where V_i is a conservative variance estimate of arm i:
            V_i = AVG([rewards_i**2]) - AVG([rewards_i])**2 + sqrt(2ln(t) / n_i)
        Nore that we don't apply the min(1/4, ...) operator to the variance because this bandit is meant for non-Bernoulli applications as well

        Returns:
            Tensor: An array of UCB scores (one per arm)
        """
        avg_rewards = self.get_avg_reward_values()
        log_t_over_ni = (
            math.log(self.total_n_obs_all_arms + 1) / self.total_n_obs_per_arm
        )
        per_arm_var_est = (
            self.total_sum_reward_squared_per_arm / self.total_n_obs_per_arm
            - avg_rewards ** 2
            + torch.sqrt(
                2 * log_t_over_ni
            )  # additional term to make the estimate conservative (unlikely to underestimate)
        )
        ucb = avg_rewards + torch.sqrt(log_t_over_ni * per_arm_var_est)
        return torch.where(
            self.total_n_obs_per_arm > 0,
            ucb,
            torch.tensor(torch.inf, dtype=torch.float),
        )


class MetricUCB(BaseUCB):
    """
    This is an improvement over UCB1 which uses a more precise confidence radius, especially for small expected rewards.
    Reference: https://arxiv.org/pdf/0809.4882.pdf
    """

    def get_ucb_scores(self):
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
        ucb = avg_rewards + torch.sqrt(avg_rewards * log_t_over_ni) + log_t_over_ni
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
