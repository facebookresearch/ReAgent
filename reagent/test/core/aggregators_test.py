#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import unittest

import torch
from reagent.core.aggregators import ActionCountAggregator


class ActionCountAggregatorTest(unittest.TestCase):
    def setUp(self):
        self.actions = ["A", "B", "C"]
        key = "logged_action"
        self.aggregator = ActionCountAggregator(key, self.actions)
        logged_actions = [
            [
                torch.tensor([0, 0, 1, 2, 2]).unsqueeze(1),
                torch.tensor([0, 1, 1, 2, 2]).unsqueeze(1),
            ],
            [
                torch.tensor([1, 0, 1, 2, 0]).unsqueeze(1),
                torch.tensor([0, 1, 1, 0, 2]).unsqueeze(1),
            ],
        ]
        for x in logged_actions:
            self.aggregator(key, x)

    def test_get_distributions(self):
        distr = self.aggregator.get_distributions()
        self.assertEqual(len(distr), 3)
        self.assertEqual(distr["A"], [0.3, 0.4])
        self.assertEqual(distr["B"], [0.3, 0.4])
        self.assertEqual(distr["C"], [0.4, 0.2])

    def test_get_cumulative_distributions(self):
        distr = self.aggregator.get_cumulative_distributions()
        self.assertEqual(len(distr), 3)
        self.assertEqual(distr["A"], 0.35)
        self.assertEqual(distr["B"], 0.35)
        self.assertEqual(distr["C"], 0.3)
