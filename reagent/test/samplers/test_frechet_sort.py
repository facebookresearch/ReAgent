#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import torch
from reagent.samplers.frechet import FrechetSort
from reagent.test.base.horizon_test_base import HorizonTestBase


class FrechetSortTest(HorizonTestBase):
    def test_log_prob(self):
        scores = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )
        shape = 2.0
        frechet_sort = FrechetSort(topk=3, shape=shape, log_scores=True)

        # The log-prob should be the same; the last 2 positions don't matter
        action = torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [1, 2, 3, 0, 4],
            ],
            dtype=torch.long,
        )
        log_probs = frechet_sort.log_prob(scores, action)
        self.assertEqual(log_probs[0], log_probs[1])

        action = torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [3, 2, 1, 0, 4],
            ],
            dtype=torch.long,
        )
        log_probs = frechet_sort.log_prob(scores, action)
        self.assertLess(log_probs[0], log_probs[1])

        # manually calculating the log prob for the second case
        s = scores[1][action[1]]
        log_prob = 0.0
        for p in range(3):
            log_prob -= torch.exp((s[p:] - s[p]) * shape).sum().log()

        self.assertAlmostEqual(log_prob, log_probs[1])

    def test_log_prob_padding(self):
        scores = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
            ],
            requires_grad=True,
        )
        shape = 2.0
        frechet_sort = FrechetSort(topk=3, shape=shape, log_scores=True)

        # A shorter sequence should have a higher prob
        action = torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [0, 1, 5, 5, 5],
            ],
            dtype=torch.long,
        )
        log_probs = frechet_sort.log_prob(scores, action)
        self.assertLess(log_probs[0], log_probs[1])

        log_probs.sum().backward()
        self.assertGreater(scores.grad.sum(), 0)

        # manually calculating the log prob for the second case
        # 5 is padding, so we remove it here
        s = scores[1][action[1][:2]]
        log_prob = 0.0
        for p in range(2):
            log_prob -= torch.exp((s[p:] - s[p]) * shape).sum().log()

        self.assertAlmostEqual(log_prob, log_probs[1])
