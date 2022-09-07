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
                reward=torch.tensor([[1.5], [2.3]], dtype=torch.float),
            ),
            CBInput(
                context_arm_features=torch.tensor(
                    [
                        [1, 4, 5, 8, 9, 4 * 8, 4 * 9, 5 * 8, 5 * 9],
                        [1, 4, 5, 12, 13, 4 * 12, 4 * 13, 5 * 12, 5 * 13],
                    ],
                    dtype=torch.float,
                ),
                reward=torch.tensor([[1.9], [2.8]], dtype=torch.float),
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

    def test_linucb_discount_factors(self) -> None:
        # change the precision to double
        torch.set_default_dtype(torch.float64)

        scorer = DisjointLinearRegressionUCB(self.num_arms, self.x_dim, gamma=0.9)
        policy = Policy(scorer=scorer, sampler=GreedyActionSampler())
        trainer = DisjointLinUCBTrainer(policy)

        # 1st round training
        trainer.training_step(self.batch, 0)
        trainer.on_train_epoch_end()

        # 2nd round training
        torch.manual_seed(0)
        self.batch_2nd_round = [
            CBInput(
                context_arm_features=torch.randn((10, self.x_dim)),
                reward=torch.randn((10, 1)),
            ),
            CBInput(
                context_arm_features=torch.randn((3, self.x_dim)),
                reward=torch.randn((3, 1)),
            ),
        ]
        self.second_batch_2nd_round = [
            CBInput(
                context_arm_features=torch.randn((10, self.x_dim)),
                reward=torch.randn((10, 1)),
            ),
            CBInput(
                context_arm_features=torch.randn((3, self.x_dim)),
                reward=torch.randn((3, 1)),
            ),
        ]
        trainer.training_step(self.batch_2nd_round, 0)
        # check if there are several training steps in a round
        # discount factor is only applied one time
        trainer.training_step(self.second_batch_2nd_round, 1)
        trainer.on_train_epoch_end()

        # eval dataset
        inp1 = torch.randn((5, self.x_dim))
        out1 = scorer(inp1)
        # check it won't do redundant coefficent update after 2nd eval
        inp2 = torch.randn((5, self.x_dim))
        out2 = scorer(inp2)

        # the feature matrix and model parameter and eval output (computed by hand)
        for arm in range(self.num_arms):
            x1 = self.batch[arm].context_arm_features.numpy()
            x2 = self.batch_2nd_round[arm].context_arm_features.numpy()
            x3 = self.second_batch_2nd_round[arm].context_arm_features.numpy()
            reward1 = self.batch[arm].reward.squeeze().numpy()
            reward2 = self.batch_2nd_round[arm].reward.squeeze().numpy()
            reward3 = self.second_batch_2nd_round[arm].reward.squeeze().numpy()

            # all matrix and vectors are the same
            A = scorer.gamma * x1.T @ x1 + x2.T @ x2 + x3.T @ x3
            b = scorer.gamma * x1.T @ reward1 + x2.T @ reward2 + x3.T @ reward3
            npt.assert_allclose(
                scorer.A[arm].numpy(), scorer.gamma * A, atol=1e-5, rtol=1e-5
            )
            npt.assert_allclose(
                scorer.b[arm].numpy(), scorer.gamma * b, atol=1e-5, rtol=1e-5
            )

            inv_A = np.linalg.inv(A + np.identity(self.x_dim) * scorer.l2_reg_lambda)
            npt.assert_allclose(scorer.inv_A[arm].numpy(), inv_A, atol=1e-4, rtol=1e-4)

            # model parameters are the same
            theta = inv_A @ b
            npt.assert_allclose(scorer.coefs[arm].numpy(), theta, atol=1e-4, rtol=1e-4)

            # ucb scores are the same
            def calculated_expected_ucb_scores(inp):
                expected_out = np.zeros(inp.size()[0])
                for i in range(inp.size()[0]):
                    x = inp[i].numpy()
                    expected_out[i] = x @ theta.T + scorer.ucb_alpha * np.sqrt(
                        x @ inv_A @ x.T
                    )
                return expected_out

            expected_out1 = calculated_expected_ucb_scores(inp1)
            npt.assert_allclose(
                out1[:, arm].numpy(), expected_out1, atol=1e-4, rtol=1e-4
            )

            expected_out2 = calculated_expected_ucb_scores(inp2)
            npt.assert_allclose(
                out2[:, arm].numpy(), expected_out2, atol=1e-4, rtol=1e-4
            )
