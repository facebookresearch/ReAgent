#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
import unittest

import numpy as np
import torch
from reagent.training.gradient_free.ars_util import ARSOptimizer


class TestARSOptimizer(unittest.TestCase):
    def metric(self, x):
        # Ackley Function
        # https://www.sfu.ca/~ssurjano/ackley.html

        x *= 100
        return (
            -20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size))
            - np.exp(np.cos(2 * np.pi * x).sum() / x.size)
            + 20
            + np.e
        )

    def test_ars_optimizer(self):
        dim = 10
        n_generations = 30
        X = torch.Tensor([[i] for i in range(dim)])
        y = torch.ones(dim)
        n_pert = 100
        feature_dim = 2
        np.random.seed(seed=123456)
        ars_opt = ARSOptimizer(feature_dim, n_pert, rand_ars_params=True)
        for i in range(n_generations):
            perturbed_params = ars_opt.sample_perturbed_params()
            rewards = []
            for idx in range(0, len(perturbed_params)):
                pos_param, neg_param = perturbed_params[idx]
                pos_weight = torch.sigmoid(
                    torch.matmul(torch.column_stack((X, y)), pos_param)
                )
                # ARSOptimizer works in an ascent manner,
                # thus a neg sign for minimizing objectives.
                r_pos = -self.metric(pos_weight.numpy())
                rewards.append(r_pos)
                neg_weight = torch.sigmoid(
                    torch.matmul(torch.column_stack((X, y)), neg_param)
                )
                r_neg = -self.metric(neg_weight.numpy())
                rewards.append(r_neg)
            ars_opt.update_ars_params(torch.Tensor(rewards))
            new_weight = torch.sigmoid(
                torch.matmul(
                    torch.column_stack((X, y)),
                    torch.from_numpy(ars_opt.ars_params).float(),
                )
            )
            perf = self.metric(new_weight.numpy())
            print(f"gen {i}: perf {perf}")
        self.assertLessEqual(perf, 1e-15)
