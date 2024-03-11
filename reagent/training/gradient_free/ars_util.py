#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

from operator import itemgetter

import numpy as np
import torch


"""
Utility functions for Advanced Random Search (ARS) algorithm
based on the paper "Simple random search provides a competitive approach
to reinforcement learning", Mania et al.
https://arxiv.org/abs/1803.07055

Here, we show an example of training a data reweighting policy using ARS. The policy
is learned to weight each sample for training a supervised learning model. ARS is a
competitive alternative to the policy gradient method in "Data Valuation using
Reinforcement Learning", Yoon, Arik, and Pfister.
https://arxiv.org/abs/1909.11671


    def reward_func(pos_param, neg_param):
        # Return rewards for positively/negatively perturbed parameters
        # model = a supervised learning model
        # X = training features
        # y = labels

        # Initialize a supervised learning model
        model_pos = model.init()
        # Sample weights are bounded within (0, 1)
        pos_weight = torch.sigmoid(torch.matmul(torch.column_stack((X, y)), pos_param))
        model_pos.fit(X, y, sample_weight=pos_weight)
        r_pos = metric(model_pos.predict(X_e), y_e)

        model_neg = model.init()
        neg_weight = torch.sigmoid(torch.matmul(torch.column_stack((X, y)), neg_param))
        model_neg.fit(X, y, sample_weight=neg_weight)
        r_neg = metric(model_neg.predict(X_e), y_e)

        return (r_pos, r_neg)

    # Training
    # feature_dim = feature dimension + 1 (for label)
    # n_pert = given number of random perturbations
    # alpha = step size
    # noise = noise level (between 0 ~ 1) added to the random perturbations
    ars_opt = ARSOptimizer(feature_dim, n_pert, alpha=alpha, noise=noise)

    for _ in range(n_generations):
        perturbed_params = ars_opt.sample_perturbed_params()
        rewards = []
        for idx in range(0, len(perturbed_params)):
            pos_param, neg_param = params[idx]
            rewards.extend(reward_func(pos_param, neg_param))
        ars_opt.update_ars_params(rewards)
"""


class ARSOptimizer:
    """ARSOptimizer is supposed to maximize an objective function"""

    def __init__(
        self,
        feature_dim,
        n_pert: int = 10,
        rand_ars_params: bool = False,
        alpha: int = 1,
        noise: int = 1,
        b_top=None,
    ) -> None:
        self.feature_dim = feature_dim
        self.ars_params = (
            np.random.randn(feature_dim) if rand_ars_params else np.zeros(feature_dim)
        )
        self.alpha = alpha
        self.noise = noise
        self.n_pert = n_pert
        self.b_top = b_top if b_top is not None else n_pert
        self.perturbations = []

    def update_ars_params(self, rewards: torch.Tensor) -> None:
        """
        reward should be something like
        [reward_pert1_pos, reward_pert1_neg, reward_pert2_pos, reward_pert2_neg, ...]
        """
        assert (
            len(self.perturbations) > 0
        ), "must call sample_perturbed_params before this function"
        assert rewards.shape == (
            2 * self.n_pert,
        ), "rewards must have length 2 * n_pert"
        rank = {}
        rewards = rewards.numpy()
        for pert_idx in range(self.n_pert):
            reward_pos = rewards[2 * pert_idx]
            reward_neg = rewards[2 * pert_idx + 1]
            rank[pert_idx] = max(reward_pos, reward_neg)
            self.perturbations[pert_idx] *= reward_pos - reward_neg
        std_r = np.std(rewards)
        weight_sum = 0
        for pert_idx in list(
            dict(sorted(rank.items(), key=itemgetter(1), reverse=True)).keys()
        )[: self.b_top]:
            weight_sum += self.perturbations[pert_idx]
        self.ars_params = self.ars_params + self.alpha * weight_sum / (
            self.b_top * (std_r if std_r > 0 else 1)
        )
        self.perturbations = []

    def sample_perturbed_params(self):
        """Return tuples of (pos_param, neg_param)"""
        self.perturbations = []
        perturbed_params = []
        for _ in range(self.n_pert):
            pert = np.random.randn(self.feature_dim)
            self.perturbations.append(pert)
            perturbed_params.append(
                (
                    torch.from_numpy(self.ars_params + self.noise * pert).float(),
                    torch.from_numpy(self.ars_params - self.noise * pert).float(),
                )
            )
        return perturbed_params
