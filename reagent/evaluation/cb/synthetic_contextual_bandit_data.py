#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import random
from dataclasses import replace
from typing import Any, Optional, Tuple

import reagent.core.types as rlt

import torch
from reagent.core.types import CBInput
from reagent.gym.policies.policy import Policy as ReAgentPolicy
from reagent.gym.policies.samplers.discrete_sampler import GreedyActionSampler

from reagent.gym.types import Sampler, Scorer

from reagent.models.linear_regression import LinearRegressionUCB
from reagent.training.cb.base_trainer import BaseCBTrainerWithEval
from reagent.training.cb.linucb_trainer import LinUCBTrainer
from reagent.training.parameters import LinUCBTrainerParameters


class Policy(ReAgentPolicy):
    def __init__(self, scorer: Scorer, sampler: Sampler) -> None:
        super().__init__(scorer=scorer, sampler=sampler)

    def act(
        self, obs: Any, possible_actions_mask: Optional[torch.Tensor] = None
    ) -> rlt.ActorOutput:
        """
        Chooses the best arm from a set of availabe arms based on contextual features.
        GreedyActionSampler may be applied.
        """
        scorer_inputs = (obs,)
        if possible_actions_mask is not None:
            scorer_inputs += (possible_actions_mask,)
        model_output = self.scorer(*scorer_inputs)
        scores = model_output["ucb"]
        actor_output = self.sampler.sample_action(scores)
        return actor_output.cpu().detach()


class DynamicBanditAgent:
    def __init__(
        self,
        trainer: BaseCBTrainerWithEval,
        feature_dim: int,
        policy=Policy,
    ):
        self._feature_dim = feature_dim
        self.trainer = trainer
        self.policy = policy

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @classmethod
    def make_agent(cls, feature_dim: int):
        """
        This method initializes the trainer, model and sampler for the agent.
        Defaultly the policy_network is LinearRegressionUCB, trainer is joint model LinUCBTrainer, and sampler is GreedyActionSampler.
        """
        params = LinUCBTrainerParameters()
        policy_network = LinearRegressionUCB(input_dim=feature_dim)
        policy = Policy(scorer=policy_network, sampler=GreedyActionSampler())
        # pyre-ignore
        trainer = LinUCBTrainer(policy, **params.asdict())
        agent = cls(
            trainer=trainer,
            feature_dim=feature_dim,
            policy=policy,
        )
        return agent

    def choose_action(self, features_all_arms) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given avaialble actions(bandit arms) and features of these actions, the agent chooses the best action.
        Args:
            features_all_arms: 3D tensor of shape (batch_size, num_arms_per_episode, feature_dim)
        What this method does:
            1) calculate the UCB score for each arm
            2) choose the best arm for each arm by finding max UCB
        """
        actor_output = self.policy.act(obs=features_all_arms)
        chosen_action = torch.argmax(actor_output.action, dim=1)
        log_prob = actor_output.log_prob
        return torch.unsqueeze(chosen_action, 1), log_prob

    def act(
        self,
        obs: CBInput,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method calls choose_action to get action based on obs
        """
        chosen_action, log_prob = self.choose_action(
            features_all_arms=obs.context_arm_features
        )
        return chosen_action, log_prob


class DynamicBanditEnv:
    """
    Design Document https://fburl.com/gdoc/0rh98o3e
    DynamicBanditEnv will interact with the DynamicBanditAgent.
    DynamicBanditAgent chooses an action. Env gives reward of this action.

    DynamicBanditEnv creates episodes where the number of bandit arms (CBInput.arms)
    is not constant across episodes, and it is much smaller than total available number of arms.
    In the application of Ads Container project, there can be 30 Millions of ads but each campaign contains only 3~30 ads.
    Here ads=arms, campaign = episode.

    Each batch contains batch_size episodes. Each episode has num_arms_per_episode arms.
    Example:
    batch_size = 2, num_arms_per_episode = 3, num_unique_batches = 4, num_arms_all = 2*3*4
    There are 24 unique arms (namely a~w):
    - [[a,b,c] [d,e,f]],  [[g,h,i] [j,k,l]],  [[m,n,o] [p,q,r]],  [[s,t,u] [v,w,x]]
    Some batch may re-appear (a group of ad campaigns reappear):
    - [[a,b,c] [d,e,f]],  [[g,h,i] [j,k,l]],  [[m,n,o] [p,q,r]],  [[s,t,u] [v,w,x]],  [[a,b,c] [d,e,f]]
    The re-appeared ads (a~f) will have similar but differetn features vs their prev appearance.
    """

    def __init__(
        self,
        num_unique_batches: int = 100,
        batch_size: int = 4,
        num_arms_per_episode: int = 10,
        feature_dim: int = 500,
        mu_shift: float = 0.0,
        sigma_shift: float = 0.0,
        reward_noise_sigma: float = 0.01,
        batch: CBInput = None,  # pyre-ignore
    ):
        self.num_unique_batches = num_unique_batches
        self._batch_size = batch_size
        self._num_arms_per_episode = num_arms_per_episode
        self._feature_dim = feature_dim
        self._num_arms_all = num_unique_batches * batch_size * num_arms_per_episode

        self.mu_shift = mu_shift
        self.sigma_shift = sigma_shift
        self.reward_noise_sigma = reward_noise_sigma

        self.gen_all_arms_ids()
        self.gen_all_arms_feature_distribution()
        self.weight = self.gen_mapping_weights()
        self.reward_shifts = self.gen_all_arms_reward_shifts()
        self.reward_regret_track_start()

    @property
    def num_arms_all(self) -> int:
        return self._num_arms_all

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_arms_per_episode(self) -> int:
        return self._num_arms_per_episode

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def reward_regret_track_start(self):
        self.accumulated_rewards = []
        self.accumulated_regrets = []
        self.accumulated_rewards_final = 0.0
        self.accumulated_regrets_final = 0.0

    def gen_all_arms_reward_shifts(self):
        """
        This method generates the reward shift_k for all arms at the beginning of this program only once.
        When a batch of arms (with sample_batch_idx) show up, index their shift_k.
        """
        reward_shifts = (
            torch.randn(self._num_arms_all) * self.sigma_shift + self.mu_shift
        )
        return reward_shifts

    def gen_mapping_weights(
        self,
    ) -> torch.Tensor:
        """
        Assume there is linear relationship between feature and reward: r_k = g(f_k) + shift_k
        This method is the weights of the linear function g().
        The weight is constant for all arms and is independent of input feature.
        The weight is a 1d tensor with feature_dim trainable varialbles.
        """
        weight = torch.randn(self.feature_dim)
        return weight

    def gen_all_arms_ids(
        self,
    ) -> None:
        """
        This generates all unique arms_ids, a 3d tensor of shape (num_unique_batches, batch_size, num_arms_per_episode)
        """
        num_all_arms = (
            self.num_unique_batches * self.batch_size * self.num_arms_per_episode
        )
        all_arms_ids = torch.randperm(num_all_arms)
        self.all_unique_arm_ids = all_arms_ids.reshape(
            [self.num_unique_batches, self.batch_size, self.num_arms_per_episode]
        )
        assert self.all_unique_arm_ids.ndim == 3
        return

    def gen_arms_per_batch(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This generates a batch of arms_ids, 1 2d tnesor of shape (batch_size, num_arms_per_episode)
        The prefix "sample_" of sample_batch_idx indicates it is a random number.
        It is not the index of a training loop such as "for idx in [0~num_batches]".
        """
        sample_batch_idx = random.randint(0, self.num_unique_batches - 1)
        arms_ids_batch = self.all_unique_arm_ids[sample_batch_idx]
        # pyre-fixme[7]: Expected `Tuple[Tensor, Tensor]` but got `Tuple[int,
        #  typing.Any]`.
        return sample_batch_idx, arms_ids_batch

    def gen_all_arms_feature_distribution(self) -> None:
        """
        Generate the distribution of feature, i.e., N(mf, sf).
        mf, sf is a 4d tensor of shape (num_unique_batches, batch_size, num_arms_per_episode, feature_dim).
        """
        zeros = torch.zeros(
            [
                self.num_unique_batches,
                self.batch_size,
                self.num_arms_per_episode,
                self.feature_dim,
            ]
        )
        ones = torch.ones(
            [
                self.num_unique_batches,
                self.batch_size,
                self.num_arms_per_episode,
                self.feature_dim,
            ]
        )
        mf = torch.distributions.Normal(zeros, ones).sample()
        sf = torch.distributions.Normal(zeros, ones).sample()
        sf = abs(sf)
        self.mf, self.sf = mf, sf
        return

    def gen_features_batch(self, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        This method get feature of each arm of a batch of arms.
        Input batch_arms is 2d tensor with shape (batch_size, num_arms_per_episode)
        Output context_arm_features is a 3d tensor with shape (batch_size, num_arms_per_episode, feature_dim)
        """
        mu = self.mf[
            batch_idx
        ]  # f_mu is 3d tensor of shape (batch_size, num_arms_per_episode, feature_dim)
        sigma = self.sf[
            batch_idx
        ]  # f_sigma is 3d tensor of shape (batch_size, num_arms_per_episode, feature_dim)
        context_arm_features = torch.distributions.Normal(mu, sigma).sample()
        return context_arm_features

    def features_to_rewards(
        self,
        inp_feature: torch.Tensor,
        sample_batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        We assume there is linear relationship between feature and reward : r_k = g(f_k) + shift_k
        The g() is a linear function mapping feature to reward
        The shift_k is a shift of reward and it follows Gaussian distribution with mean 0 and variance sigma_shift
        The output is reward, a float number
        """
        # map features to rewards through a linear function W
        reward = torch.nn.functional.linear(
            input=inp_feature, weight=self.weight, bias=None
        )

        # add some Gaussian noise to each reward
        shift_k = self.reward_shifts[sample_batch_idx]
        reward += shift_k
        noise_k = torch.randn_like(reward) * self.reward_noise_sigma
        reward += noise_k
        return reward

    def get_batch(self) -> CBInput:
        """
        Each batch's feature is of shape (batch_size, num_arms_per_episode, feature_dim), a 3d tensor.
        Each arm is associated with its feature (CBInput.context_arm_features).
        Each arm's feature is mapped to a reward by a linear function.
        The rewards of all arms is added to the batch.
        """
        sample_batch_idx, batch_arms = self.gen_arms_per_batch()
        context_arm_features = self.gen_features_batch(batch_idx=sample_batch_idx)
        assert context_arm_features.ndim == 3
        rewards_all_arms = self.features_to_rewards(
            inp_feature=context_arm_features, sample_batch_idx=sample_batch_idx
        )
        batch = CBInput(
            context_arm_features=context_arm_features,
            arms=batch_arms,  # ads of batch_size campaigns
            rewards_all_arms=rewards_all_arms,
        )
        return batch

    def add_chosen_action_reward(self, chosen_action_idx, batch) -> CBInput:
        """
        The agent provides the chosen action, and
        the env adss the chosen action to the batch/CBInput.
        """
        assert batch.rewards_all_arms.shape == (
            self.batch_size,
            self.num_arms_per_episode,
        )
        chosen_reward = batch.rewards_all_arms.gather(1, chosen_action_idx)
        new_batch = replace(batch, reward=chosen_reward, action=chosen_action_idx)
        assert new_batch.action.shape == (self.batch_size, 1)
        assert chosen_reward.shape == (self.batch_size, 1)

        self.reward_regret_tracking(chosen_reward=chosen_reward, batch=batch)
        return new_batch

    def reward_regret_tracking(self, chosen_reward, batch):
        self.accumulated_rewards_final += chosen_reward.sum().item()
        self.accumulated_rewards.append(self.accumulated_rewards_final)
        chosen_regret = torch.max(batch.rewards_all_arms, dim=1).values - torch.squeeze(
            chosen_reward
        )
        self.accumulated_regrets_final += chosen_regret.sum().item()
        self.accumulated_regrets.append(self.accumulated_regrets_final)
