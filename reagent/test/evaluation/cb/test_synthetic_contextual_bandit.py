#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import torch
from reagent.evaluation.cb.run_synthetic_bandit import run_dynamic_bandit_env


class TestSyntheticBandit(unittest.TestCase):
    """
    Test run of the DynamitBandits environment and agent.
    Check the coefs of the model has been updated.
    Get the accumulated rewards and accumulated regrets.
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

    def test_run_synthetic_bandit(
        self,
        feature_dim: int = 3,
        num_unique_batches: int = 10,
        batch_size: int = 4,
        num_arms_per_episode: int = 2,
        num_obs: int = 101,
        max_epochs: int = 1,
    ):
        agent, accumulated_rewards, accumulated_regrets = run_dynamic_bandit_env(
            feature_dim=feature_dim,
            num_unique_batches=num_unique_batches,
            batch_size=batch_size,
            num_arms_per_episode=num_arms_per_episode,
            num_obs=num_obs,
            max_epochs=max_epochs,
        )
        coefs_post_train = agent.trainer.scorer.avg_A
        assert torch.count_nonzero(coefs_post_train) > 0
        assert accumulated_regrets[-1] >= accumulated_regrets[0]

        accumulated_regrets_tensor = torch.tensor(accumulated_regrets)
        accumulated_rewards_tensor = torch.tensor(accumulated_rewards)
        assert torch.all(
            accumulated_regrets_tensor[1:] >= accumulated_regrets_tensor[:-1]
        )
        assert accumulated_rewards_tensor.shape == (num_obs,)
        assert accumulated_regrets_tensor.shape == (num_obs,)
        return accumulated_rewards_tensor, accumulated_regrets_tensor


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
