#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import copy
import unittest

import numpy as np
import numpy.testing as npt
import torch
from reagent.core.types import CBInput
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.discrete_sampler import GreedyActionSampler
from reagent.models.disjoint_linucb_predictor import DisjointLinearRegressionUCB
from reagent.training.cb.disjoint_linucb_trainer import DisjointLinUCBTrainer
from reagent.training.parameters import DisjointLinUCBTrainerParameters


class TestDisjointLinUCB(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_arms = 2
        self.params = DisjointLinUCBTrainerParameters()

        self.x_dim = 9
        policy_network = DisjointLinearRegressionUCB(self.num_arms, self.x_dim)
        self.policy = Policy(scorer=policy_network, sampler=GreedyActionSampler())

        self.trainer = DisjointLinUCBTrainer(self.policy, **self.params.asdict())
        self.batch = [
            CBInput(
                context_arm_features=torch.tensor(
                    [
                        [1, 2, 3, 6, 7, 2 * 6, 2 * 7, 3 * 6, 3 * 7],
                        [1, 2, 3, 10, 11, 2 * 10, 2 * 11, 3 * 10, 3 * 11],
                    ],
                    dtype=torch.float,
                ),
                reward=torch.tensor([[1.5], [2.3]]),
            ),
            CBInput(
                context_arm_features=torch.tensor(
                    [
                        [1, 4, 5, 8, 9, 4 * 8, 4 * 9, 5 * 8, 5 * 9],
                        [1, 4, 5, 12, 13, 4 * 12, 4 * 13, 5 * 12, 5 * 13],
                    ],
                    dtype=torch.float,
                ),
                reward=torch.tensor([[1.9], [2.8]]),
            ),
        ]

    def test_linucb_training_step(self):
        self.trainer.training_step(self.batch, 0)

    def test_linucb_training_batch_vs_online(self):
        # make sure that feeding in a batch gives same result as feeding in examples one-by-one
        obss = [[], []]
        for i in range(self.batch_size):
            obss[i].append(
                CBInput(
                    context_arm_features=self.batch[0].context_arm_features[
                        i : i + 1, :
                    ],
                    reward=self.batch[0].reward[[i]],
                )
            )
            obss[i].append(
                CBInput(
                    context_arm_features=self.batch[1].context_arm_features[
                        i : i + 1, :
                    ],
                    reward=self.batch[1].reward[[i]],
                )
            )

        scorer_1 = DisjointLinearRegressionUCB(self.num_arms, self.x_dim)
        scorer_2 = DisjointLinearRegressionUCB(self.num_arms, self.x_dim)
        policy_1 = Policy(scorer=scorer_1, sampler=GreedyActionSampler())
        policy_2 = Policy(scorer=scorer_2, sampler=GreedyActionSampler())
        trainer_1 = DisjointLinUCBTrainer(policy_1)
        trainer_2 = DisjointLinUCBTrainer(policy_2)

        trainer_1.training_step(obss[0], 0)
        trainer_1.training_step(obss[1], 1)
        trainer_2.training_step(self.batch, 0)

        for arm in range(self.num_arms):
            npt.assert_array_less(
                np.zeros(scorer_1.A[arm].shape), scorer_1.A[arm].numpy()
            )  # make sure A got updated
            npt.assert_allclose(
                scorer_1.A[arm].numpy(), scorer_2.A[arm].numpy(), rtol=1e-4
            )
            npt.assert_allclose(
                scorer_1.inv_A[arm].numpy(), scorer_2.inv_A[arm].numpy(), rtol=1e-4
            )
            npt.assert_allclose(
                scorer_1.b[arm].numpy(), scorer_2.b[arm].numpy(), rtol=1e-4
            )

    def test_linucb_model_update_equations(self):
        # make sure that the model parameters match hand-computed values
        scorer = DisjointLinearRegressionUCB(self.num_arms, self.x_dim)
        policy = Policy(scorer=scorer, sampler=GreedyActionSampler())
        trainer = DisjointLinUCBTrainer(policy)
        trainer.training_step(self.batch, 0)
        # the feature matrix (computed by hand)
        for arm in range(self.num_arms):
            x = self.batch[arm].context_arm_features.numpy()
            npt.assert_allclose(scorer.A[arm].numpy(), x.T @ x, rtol=1e-5)
            npt.assert_allclose(
                scorer.b[arm].numpy(),
                x.T @ self.batch[arm].reward.squeeze().numpy(),
                rtol=1e-5,
            )

        scorer._estimate_coefs()
        for arm in range(self.num_arms):
            npt.assert_allclose(
                (np.eye(self.x_dim) + scorer.A[arm].numpy())
                @ scorer.inv_A[arm].numpy(),
                np.eye(self.x_dim),
                atol=1e-3,
            )
            npt.assert_equal(
                scorer.A[arm].numpy(), scorer.coefs_valid_for_A[arm].numpy()
            )

    def test_linucb_weights(self):
        # make sure that using a weight is same as processing an example several times
        batch_with_weight = copy.deepcopy(self.batch)
        for arm in range(self.num_arms):
            batch_with_weight[arm].weight = 3 * torch.ones((self.batch_size, 1))

        scorer_1 = DisjointLinearRegressionUCB(self.num_arms, self.x_dim)
        scorer_2 = DisjointLinearRegressionUCB(self.num_arms, self.x_dim)
        policy_1 = Policy(scorer=scorer_1, sampler=GreedyActionSampler())
        policy_2 = Policy(scorer=scorer_2, sampler=GreedyActionSampler())
        trainer_1 = DisjointLinUCBTrainer(policy_1)
        trainer_2 = DisjointLinUCBTrainer(policy_2)

        trainer_1.training_step(batch_with_weight, 0)
        for i in range(3):
            trainer_2.training_step(self.batch, i)

        for arm in range(self.num_arms):
            npt.assert_array_less(
                np.zeros(scorer_1.A[arm].shape), scorer_1.A[arm].numpy()
            )  # make sure A got updated
        npt.assert_allclose(scorer_1.A.numpy(), scorer_2.A.numpy(), rtol=1e-6)
        npt.assert_allclose(scorer_1.b.numpy(), scorer_2.b.numpy(), rtol=1e-6)
