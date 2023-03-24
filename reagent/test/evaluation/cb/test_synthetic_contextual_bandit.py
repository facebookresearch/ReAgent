#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import torch
from reagent.evaluation.cb.run_synthetic_bandit import run_dynamic_bandit_env


class TestSyntheticBandit(unittest.TestCase):
    """
    Test run of the DynamitBandits environment and agent.
    Check the coefs of the model has been updated.
    """

    def setUp(
        self,
        feature_dim: int = 3,
        num_unique_batches: int = 10,
        batch_size: int = 4,
        num_arms_per_episode: int = 2,
    ):
        self.feature_dim = feature_dim
        self.num_unique_batches = num_unique_batches
        self.batch_size = batch_size
        self.num_arms_per_episode = num_arms_per_episode

    def test_run_synthetic_bandit(self):
        agent = run_dynamic_bandit_env(
            feature_dim=3,
            num_unique_batches=10,
            batch_size=4,
            num_arms_per_episode=2,
            num_obs=101,
            max_epochs=1,
        )
        coefs_post_train = agent.trainer.scorer.avg_A
        assert torch.count_nonzero(coefs_post_train) > 0


"""
To run on Bento :
    test = TestSyntheticBandit()
    test.setUp()
    test.test_run_synthetic_bandit()

To run on Devserver :
    buck test reagent:evaluation_tests -- TestSyntheticBandit
    OR
    buck2 test @mode/dev //reagent:evaluation_tests -- --exact 'reagent:evaluation_tests - test_run_synthetic_bandit (reagent.test.evaluation.cb.test_synthetic_contextual_bandit.TestSyntheticBandit)'
"""
