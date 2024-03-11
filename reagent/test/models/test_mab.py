#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import unittest
from dataclasses import replace

import numpy as np
import numpy.testing as npt
import torch
from reagent.core.types import CBInput
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.discrete_sampler import GreedyActionSampler
from reagent.models.mab import UCB1MAB
from reagent.training.cb.mab_trainer import MABTrainer


class TestUCB1MAB(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.batch = CBInput.from_dict(
            {
                "context_arm_features": torch.zeros(2, 3, 1),
                "action": torch.Tensor([[2], [0]]).long(),
                "reward": torch.Tensor([[3.0], [4.0]]),
                "arms": torch.Tensor([[1, 2, 3], [4, 5, 6]]).long(),
            }
        )

    def test_call(self) -> None:
        bandit = UCB1MAB(ucb_alpha=1.0)
        policy = Policy(
            scorer=bandit,
            sampler=GreedyActionSampler(),
        )
        trainer = MABTrainer(policy)

        trainer.cb_training_step(self.batch, 0, 0)

        pred_arm_ids = torch.tensor([[1, 3], [6, 4], [5, 2]]).long()
        alpha = 1.5
        model_output = bandit(pred_arm_ids, ucb_alpha=alpha)

        ucb = model_output["ucb"]
        self.assertIsInstance(ucb, torch.Tensor)
        self.assertEqual(tuple(ucb.shape), (3, 2))
        # check that we return inf for arms which we haven't observed
        self.assertEqual(min(ucb.numpy()[[0, 1, 2, 2], [0, 0, 0, 1]]), float("inf"))
        # check that we return non-inf for arms which we have observed
        self.assertLess(max(ucb.numpy()[0:1, 1]), float("inf"))

        mu = model_output["pred_label"]
        self.assertIsInstance(ucb, torch.Tensor)
        self.assertEqual(tuple(ucb.shape), (3, 2))
        npt.assert_almost_equal(
            mu.numpy(),
            np.array([[0.0, 3.0], [0.0, 4.0], [0.0, 0.0]]),
        )

        sigma = model_output["pred_sigma"]
        self.assertIsInstance(sigma, torch.Tensor)
        self.assertEqual(tuple(sigma.shape), (3, 2))

    def test_default_score(self) -> None:
        # test custom default score
        default_score = 56.3
        bandit = UCB1MAB(default_score=default_score)
        policy = Policy(
            scorer=bandit,
            sampler=GreedyActionSampler(),
        )
        trainer = MABTrainer(policy)
        trainer.cb_training_step(self.batch, 0, 0)

        pred_arm_ids = torch.tensor([[1, 3], [6, 4], [5, 2]]).long()
        model_output = bandit(pred_arm_ids)

        ucb = model_output["ucb"].numpy()
        # check that we return default_score for arms which we haven't observed
        self.assertAlmostEqual(
            np.unique(ucb[[0, 1, 2, 2], [0, 0, 0, 1]])[0], default_score, places=5
        )

        self.assertFalse(default_score in set(ucb[0:1, 1]))

    def test_min_explore_steps(self) -> None:
        # test min number of exploration steps
        min_explore_steps = 3
        bandit = UCB1MAB(min_explore_steps=min_explore_steps)
        policy = Policy(
            scorer=bandit,
            sampler=GreedyActionSampler(),
        )
        trainer = MABTrainer(policy)
        batch = replace(self.batch, weight=torch.Tensor([[2.9], [3.1]]))
        trainer.cb_training_step(batch, 0, 0)

        pred_arm_ids = torch.tensor([[3, 4]]).long()
        model_output = bandit(pred_arm_ids)

        ucb = model_output["ucb"].numpy()
        # check that we return default_score (inf) for arm which wasn't sufficiently explored
        self.assertEqual(ucb[0, 0], float("inf"))

        # check that we return finite for arm which was sufficiently explored
        self.assertLess(ucb[0, 1], float("inf"))
