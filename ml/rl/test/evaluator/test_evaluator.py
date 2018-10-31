#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest
from typing import NamedTuple

from ml.rl.training.evaluator import Evaluator


logger = logging.getLogger(__name__)


class MockSample(NamedTuple):
    mdp_id: str
    sequence_number: int
    reward: float


class TestEvaluator(unittest.TestCase):
    def test_compute_episode_value_from_samples(self):
        samples = [
            MockSample("1", 3, 1),
            MockSample("1", 5, 2),
            MockSample("1", 6, 1),
            MockSample("3", 10, 2),
            MockSample("3", 11, 1),
            MockSample("6", 2, 3),
            MockSample("6", 4, 2),
            MockSample("6", 5, 0),
            MockSample("6", 8, 1),
        ]
        logged_values = Evaluator.compute_episode_value_from_samples(samples, 0.5)
        expected_values = [1.625, 2.5, 1, 2.5, 1, 3.515625, 2.0625, 0.125, 1]

        for i, val in enumerate(logged_values):
            self.assertEquals(val, expected_values[i])
