from abc import abstractmethod
from typing import Union, Optional, List

import torch
from reagent.mab.mab_algorithm import MABAlgo, get_arm_indices, place_values_at_indices
from torch import Tensor


class BaseThompsonSampling(MABAlgo):
    @abstractmethod
    def _get_posterior_samples(self) -> Tensor:
        pass

    def forward(self):
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
    We use poterior update equations from
        https://en.wikipedia.org/wiki/Normal-gamma_distribution#Posterior_distribution_of_the_parameters
    """

    def __init__(
        self,
        *,
        n_arms: Optional[int] = None,
        arm_ids: Optional[List[Union[str, int]]] = None,
    ):
        super().__init__(n_arms=n_arms, arm_ids=arm_ids)
        self.mus = torch.zeros(self.n_arms)
        self.alpha_0 = 1.5  # initial value of the alpha parameter
        self.lambda_0 = 1.0  # initial value of the lambda parameter
        self.gamma_rates = torch.ones(self.n_arms)

    def add_single_observation(self, arm_id: int, reward: float):
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
        arm_ids: Optional[List[Union[str, int]]] = None,
    ):
        if arm_ids is None or arm_ids == self.arm_ids:
            # assume that the observations are for all arms in the default order
            arm_ids = self.arm_ids
            arm_idxs = list(range(self.n_arms))
        else:
            assert len(arm_ids) == len(
                set(arm_ids)
            )  # make sure no duplicates in arm IDs

            # get the indices of the arms
            arm_idxs = get_arm_indices(self.arm_ids, arm_ids)

            # put elements from the batch in the positions specified by `arm_ids` (missing arms will be zero)
            n_obs_per_arm = place_values_at_indices(
                n_obs_per_arm, arm_idxs, self.n_arms
            )
            sum_reward_per_arm = place_values_at_indices(
                sum_reward_per_arm, arm_idxs, self.n_arms
            )
            sum_reward_squared_per_arm = place_values_at_indices(
                sum_reward_squared_per_arm, arm_idxs, self.n_arms
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
        self.total_n_obs_per_arm += n_obs_per_arm
        self.total_sum_reward_per_arm += sum_reward_per_arm
        self.total_sum_reward_squared_per_arm += sum_reward_squared_per_arm
        self.total_n_obs_all_arms += int(n_obs_per_arm.sum())

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
