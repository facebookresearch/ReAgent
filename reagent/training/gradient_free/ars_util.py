from operator import itemgetter

import numpy as np
import torch


"""
Utility functions for Advanced Random Search algorithm
based on the paper "Simple random search provides a competitive approach
to reinforcement learning", Mania et al.
https://arxiv.org/pdf/1803.07055.pdf

Usage example:
    n_pert = given number of random perturbations
    alpha = step size
    feature_dim = feature dimension + 1 (for label)
    noise = noise level (<1 and >0) added to the random perturbations
    model = the target model
    X = training features
    y = labels
    X_e = eval features
    y_e = eval labels
    metric = eval metric

    ars_opt = ARSOptimizer(feature_dim, n_pert, alpha=alpha, noise=noise)

    for _ in range(n_generations):
        perturbed_params = ars_opt.sample_perturbed_params()
        rewards = []
        for idx in range(0, len(perturbed_params)):
            pos_param, neg_param = params[idx]
            model_pos = model.init()
            pos_weight = torch.sigmoid(torch.matmul(torch.column_stack((X, y)), pos_param))
            model_pos.fit(X, y, sample_weight=pos_weight)
            r_pos = metric(model_pos.predict(X_e), y_e)
            rewards.append(r_pos)

            model_neg = model.init()
            neg_weight = torch.sigmoid(torch.matmul(torch.column_stack((X, y)), neg_param))
            model_neg.fit(X, y, sample_weight=neg_weight)
            r_neg = metric(model_neg.predict(X_e), y_e)
            rewards.append(r_neg)
        ars_opt.update_ars_params(rewards)

    model_eval = model.init()
    eval_weight = torch.sigmoid(torch.matmul(torch.column_stack((X, y)),
                        torch.from_numpy(ars_opt.ars_params).float()))
    model_eval.fit(X, y, sample_weight=eval_weight)
    reward = metric(model_eval.predict(X_e), y_e)
"""


class ARSOptimizer:
    """ARSOptimizer is supposed to maximize an objective function"""

    def __init__(
        self,
        feature_dim,
        n_pert=10,
        rand_ars_params=False,
        alpha=1,
        noise=1,
        b_top=None,
    ):
        self.feature_dim = feature_dim
        self.ars_params = (
            np.random.randn(feature_dim) if rand_ars_params else np.zeros(feature_dim)
        )
        self.alpha = alpha
        self.noise = noise
        self.n_pert = n_pert
        self.b_top = b_top if b_top is not None else n_pert
        self.perturbations = []

    def update_ars_params(self, rewards: torch.Tensor):
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
