#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
import torch
from reagent.gym.policies.samplers.discrete_sampler import EpsilonGreedyActionSampler
from reagent.test.base.horizon_test_base import HorizonTestBase


class EpsilonGreedyActionSamplerTest(HorizonTestBase):
    def test_greedy_selection(self):
        scores = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )
        sampler = EpsilonGreedyActionSampler(epsilon=0.0)

        test_action = torch.tensor(
            [
                [0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        action = sampler.sample_action(scores)

        torch.testing.assert_allclose(action.action, test_action)

        test_log_prob = torch.tensor(
            [0.0, 0.0],
            dtype=torch.float,
        )

        torch.testing.assert_allclose(action.log_prob, test_log_prob)

    def test_uniform_random_selection(self):
        scores = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )
        sampler = EpsilonGreedyActionSampler(epsilon=1.0)

        action = sampler.sample_action(scores)

        test_log_prob = torch.tensor(
            [-1.60944, -1.60944],
            dtype=torch.float,
        )

        torch.testing.assert_allclose(action.log_prob, test_log_prob)
