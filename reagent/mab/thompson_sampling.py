#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# noqa
from abc import abstractmethod
from typing import Optional, List

import torch
from reagent.mab.mab_algorithm import MABAlgo, reindex_multiple_tensors
from torch import Tensor


class BaseThompsonSampling(MABAlgo):
    @abstractmethod
    def _get_posterior_samples(self) -> Tensor:
        pass

    def get_scores(self):
        return self._get_posterior_samples()


class BernoulliBetaThompson(BaseThompsonSampling):
    """
    The Thompson Sampling MAB with Bernoulli-Beta distribution for rewards.
    Appropriate for MAB with Bernoulli rewards (e.g CTR)
    """

    def _get_posterior_samples(self) -> Tensor:
        """
        Get samples from the posterior distributions of arm rewards
        """
        return torch.distributions.beta.Beta(
            1 + self.total_sum_reward_per_arm,
            1 + self.total_n_obs_per_arm - self.total_sum_reward_per_arm,
        ).sample()


class NormalGammaThompson(BaseThompsonSampling):
    """
    The Thompson Sampling MAB with Normal-Gamma distribution for rewards.
    Appropriate for MAB with normally distributed rewards.
    We use posterior update equations from
        https://en.wikipedia.org/wiki/Normal-gamma_distribution#Posterior_distribution_of_the_parameters
    """

    def __init__(
        self,
        randomize_ties: bool = True,
        min_num_obs_per_arm: int = 1,
        *,
        n_arms: Optional[int] = None,
        arm_ids: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            randomize_ties=randomize_ties,
            n_arms=n_arms,
            arm_ids=arm_ids,
            min_num_obs_per_arm=min_num_obs_per_arm,
        )
        self.mus = torch.zeros(self.n_arms)
        self.alpha_0 = 1.5  # initial value of the alpha parameter
        self.lambda_0 = 1.0  # initial value of the lambda parameter
        self.gamma_rates = torch.ones(self.n_arms)

    def add_single_observation(self, arm_id: str, reward: float) -> None:
        super().add_single_observation(arm_id=arm_id, reward=reward)
        arm_idx = self.arm_ids.index(arm_id)
        lambda_ = (
            self.lambda_0 + self.total_n_obs_per_arm[arm_idx] - 1
        )  # -1 bcs counter is already incremented by super() call
        self.gamma_rates[arm_idx] += (
            0.5 * (reward - self.mus[arm_idx]) ** 2 * lambda_ / (lambda_ + 1)
        )
        self.mus[arm_idx] += (reward - self.mus[arm_idx]) / (lambda_ + 1)

    def add_batch_observations(
        self,
        n_obs_per_arm: Tensor,
        sum_reward_per_arm: Tensor,
        sum_reward_squared_per_arm: Tensor,
        arm_ids: Optional[List[str]] = None,
    ) -> None:
        (
            n_obs_per_arm,
            sum_reward_per_arm,
            sum_reward_squared_per_arm,
        ) = reindex_multiple_tensors(
            all_ids=self.arm_ids,
            batch_ids=arm_ids,
            value_tensors=(
                n_obs_per_arm,
                sum_reward_per_arm,
                sum_reward_squared_per_arm,
            ),
        )

        mean_rewards_batch = torch.nan_to_num(
            sum_reward_per_arm / n_obs_per_arm, nan=0.0
        )
        lambdas = self.lambda_0 + self.total_n_obs_per_arm
        self.gamma_rates += 0.5 * n_obs_per_arm * lambdas / (
            n_obs_per_arm + lambdas
        ) * (mean_rewards_batch - self.mus) ** 2 + 0.5 * (
            sum_reward_squared_per_arm - n_obs_per_arm * mean_rewards_batch ** 2
        )
        self.mus += (sum_reward_per_arm - n_obs_per_arm * self.mus) / (
            n_obs_per_arm + lambdas
        )
        super().add_batch_observations(
            n_obs_per_arm=n_obs_per_arm,
            sum_reward_per_arm=sum_reward_per_arm,
            sum_reward_squared_per_arm=sum_reward_squared_per_arm,
            arm_ids=self.arm_ids,  # pass self.arm_ids instead of arm_ids because we've already reindexed all tensors
        )

    def _get_posterior_samples(self) -> Tensor:
        """
        Get samples from the posterior distributions of arm rewards
        """
        precisions = (
            self.lambda_0 + self.total_n_obs_per_arm
        ) * torch.distributions.gamma.Gamma(
            0.5 * (self.total_n_obs_per_arm + self.alpha_0), self.gamma_rates
        ).sample()
        return torch.distributions.normal.Normal(
            self.mus, 1.0 / torch.sqrt(precisions)
        ).sample()
