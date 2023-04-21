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
from reagent.models.linear_regression import LinearRegressionUCB
from reagent.training.cb.base_trainer import _add_chosen_arm_features
from reagent.training.cb.linucb_trainer import LinUCBTrainer
from reagent.training.parameters import LinUCBTrainerParameters


class TestLinUCButils(unittest.TestCase):
    def test_add_chosen_arm_features(self):
        all_arms_features = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float
        )
        actions = torch.tensor([[1], [0]], dtype=torch.long)
        batch = CBInput(
            context_arm_features=all_arms_features,
            action=actions,
        )
        new_batch = _add_chosen_arm_features(batch)
        npt.assert_equal(
            new_batch.features_of_chosen_arm.numpy(), np.array([[3.0, 4.0], [5.0, 6.0]])
        )


class TestLinUCB(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2

        self.num_arms = 2
        self.params = LinUCBTrainerParameters()

        self.x_dim = 5
        policy_network = LinearRegressionUCB(self.x_dim)
        self.policy = Policy(scorer=policy_network, sampler=GreedyActionSampler())

        self.trainer = LinUCBTrainer(self.policy, **self.params.asdict())
        self.batch = CBInput(
            context_arm_features=torch.tensor(
                [
                    [
                        [1, 2, 3, 6, 7],
                        [1, 2, 3, 10, 11],
                    ],
                    [
                        [1, 4, 5, 8, 9],
                        [1, 4, 5, 12, 13],
                    ],
                ],
                dtype=torch.float,
            ),
            action=torch.tensor([[0], [1]], dtype=torch.long),
            reward=torch.tensor([[1.5], [2.3]], dtype=torch.float),
        )

    def test_linucb_training_step(self):
        self.trainer.training_step(self.batch, 0)
        self.trainer.on_train_epoch_end()

    def test_linucb_training_batch_vs_online(self):
        # make sure that feeding in a batch gives same result as feeding in examples one-by-one
        obss = []
        for i in range(self.batch_size):
            obss.append(
                CBInput(
                    context_arm_features=self.batch.context_arm_features[
                        i : i + 1, :, :
                    ],
                    action=self.batch.action[[i]],
                    reward=self.batch.reward[[i]],
                )
            )

        scorer_1 = LinearRegressionUCB(self.x_dim)
        scorer_2 = LinearRegressionUCB(self.x_dim)
        policy_1 = Policy(scorer=scorer_1, sampler=GreedyActionSampler())
        policy_2 = Policy(scorer=scorer_2, sampler=GreedyActionSampler())
        trainer_1 = LinUCBTrainer(policy_1)
        trainer_2 = LinUCBTrainer(policy_2)

        trainer_1.training_step(obss[0], 0)
        trainer_1.training_step(obss[1], 1)
        trainer_1.on_train_epoch_end()
        trainer_2.training_step(self.batch, 0)
        trainer_2.on_train_epoch_end()

        npt.assert_array_less(
            np.zeros(scorer_1.avg_A.shape), scorer_1.avg_A.numpy()
        )  # make sure A got updated
        npt.assert_allclose(scorer_1.avg_A.numpy(), scorer_2.avg_A.numpy(), rtol=1e-4)
        npt.assert_allclose(scorer_1.avg_b.numpy(), scorer_2.avg_b.numpy(), rtol=1e-4)

    def test_linucb_training_multiple_epochs(self):
        # make sure that splitting the data across multiple epochs is same as learning from all data in one epoch
        # this is only true when there is no discounting (gamma=1)
        obss = []
        for i in range(self.batch_size):
            obss.append(
                CBInput(
                    context_arm_features=self.batch.context_arm_features[
                        i : i + 1, :, :
                    ],
                    action=self.batch.action[[i]],
                    reward=self.batch.reward[[i]],
                )
            )

        scorer_1 = LinearRegressionUCB(self.x_dim)
        scorer_2 = LinearRegressionUCB(self.x_dim)
        policy_1 = Policy(scorer=scorer_1, sampler=GreedyActionSampler())
        policy_2 = Policy(scorer=scorer_2, sampler=GreedyActionSampler())
        trainer_1 = LinUCBTrainer(policy_1)
        trainer_2 = LinUCBTrainer(policy_2)

        trainer_1.training_step(obss[0], 0)
        trainer_1.on_train_epoch_end()
        trainer_1.training_step(obss[1], 1)
        trainer_1.on_train_epoch_end()

        trainer_2.training_step(self.batch, 0)
        trainer_2.on_train_epoch_end()

        npt.assert_array_less(
            np.zeros(scorer_1.avg_A.shape), scorer_1.avg_A.numpy()
        )  # make sure A got updated
        npt.assert_allclose(scorer_1.avg_A.numpy(), scorer_2.avg_A.numpy(), rtol=1e-4)
        npt.assert_allclose(scorer_1.avg_b.numpy(), scorer_2.avg_b.numpy(), rtol=1e-4)
        npt.assert_allclose(
            scorer_1.inv_avg_A.numpy(), scorer_2.inv_avg_A.numpy(), rtol=1e-4
        )
        npt.assert_allclose(scorer_1.coefs.numpy(), scorer_2.coefs.numpy(), rtol=1e-3)

    def test_linucb_model_update_equations(self):
        # make sure that the model parameters match hand-computed values
        scorer = LinearRegressionUCB(self.x_dim)
        policy = Policy(scorer=scorer, sampler=GreedyActionSampler())
        trainer = LinUCBTrainer(policy)
        trainer.training_step(self.batch, 0)
        trainer.on_train_epoch_end()
        # the feature matrix (computed by hand)
        x = _add_chosen_arm_features(self.batch).features_of_chosen_arm.numpy()

        npt.assert_allclose(scorer.avg_A.numpy(), x.T @ x / len(self.batch), rtol=1e-4)
        npt.assert_allclose(
            scorer.avg_b.numpy(),
            x.T @ self.batch.reward.squeeze().numpy() / len(self.batch),
            rtol=1e-4,
        )

        scorer._calculate_coefs()
        npt.assert_equal(scorer.avg_A.numpy(), scorer.coefs_valid_for_avg_A.numpy())

        npt.assert_allclose(
            (
                np.eye(self.x_dim) * scorer.l2_reg_lambda
                + (scorer.avg_A * scorer.sum_weight).numpy()
            )
            @ (scorer.inv_avg_A / scorer.sum_weight).numpy(),
            np.eye(self.x_dim),
            atol=1e-3,
        )

    def test_linucb_weights(self):
        # make sure that using a weight is same as processing an example several times
        batch_with_weight = copy.deepcopy(self.batch)
        batch_with_weight.weight = 3 * torch.ones((self.batch_size, 1))

        scorer_1 = LinearRegressionUCB(self.x_dim)
        scorer_2 = LinearRegressionUCB(self.x_dim)
        policy_1 = Policy(scorer=scorer_1, sampler=GreedyActionSampler())
        policy_2 = Policy(scorer=scorer_2, sampler=GreedyActionSampler())
        trainer_1 = LinUCBTrainer(policy_1)
        trainer_2 = LinUCBTrainer(policy_2)

        trainer_1.training_step(batch_with_weight, 0)
        trainer_1.on_train_epoch_end()
        for i in range(3):
            trainer_2.training_step(self.batch, i)
        trainer_2.on_train_epoch_end()

        npt.assert_array_less(
            np.zeros(scorer_1.avg_A.shape), scorer_1.avg_A.numpy()
        )  # make sure A got updated
        npt.assert_allclose(scorer_1.avg_A.numpy(), scorer_2.avg_A.numpy(), rtol=1e-4)
        npt.assert_allclose(scorer_1.avg_b.numpy(), scorer_2.avg_b.numpy(), rtol=1e-4)

    def test_linucb_discount_factors(self) -> None:
        # change the precision to double
        torch.set_default_dtype(torch.float64)

        gamma = 0.8
        scorer = LinearRegressionUCB(self.x_dim, gamma=gamma)
        policy = Policy(scorer=scorer, sampler=GreedyActionSampler())
        trainer = LinUCBTrainer(policy)

        # 1st round training
        trainer.training_step(self.batch, 0)
        trainer.on_train_epoch_end()

        # 2nd round training
        torch.manual_seed(0)
        self.batch_2nd_round = CBInput(
            context_arm_features=torch.randn((10, self.num_arms, self.x_dim)),
            reward=torch.randn((10, 1)),
            action=torch.tensor([[0], [1]], dtype=torch.long).repeat(5, 1),
        )
        self.second_batch_2nd_round = CBInput(
            context_arm_features=torch.randn((6, self.num_arms, self.x_dim)),
            reward=torch.randn((6, 1)),
            action=torch.tensor([[0], [1]], dtype=torch.long).repeat(3, 1),
        )
        trainer.training_step(self.batch_2nd_round, 0)
        # check if there are several training steps in a round
        # discount factor is only applied one time
        trainer.training_step(self.second_batch_2nd_round, 1)
        trainer.on_train_epoch_end()

        # eval dataset
        inp1 = torch.randn((5, self.x_dim))
        model_output1 = scorer(inp1)
        out1 = model_output1["ucb"]
        # check it won't do redundant coefficent update after 2nd eval
        inp2 = torch.randn((5, self.x_dim))
        model_output2 = scorer(inp2)
        out2 = model_output2["ucb"]

        # the feature matrix and model parameter and eval output (computed by hand)
        x1 = _add_chosen_arm_features(self.batch).features_of_chosen_arm.numpy()
        x2 = _add_chosen_arm_features(
            self.batch_2nd_round
        ).features_of_chosen_arm.numpy()
        x3 = _add_chosen_arm_features(
            self.second_batch_2nd_round
        ).features_of_chosen_arm.numpy()
        reward1 = self.batch.reward.squeeze().numpy()
        reward2 = self.batch_2nd_round.reward.squeeze().numpy()
        reward3 = self.second_batch_2nd_round.reward.squeeze().numpy()

        # all matrix and vectors are the same
        A = scorer.gamma * x1.T @ x1 + x2.T @ x2 + x3.T @ x3
        b = scorer.gamma * x1.T @ reward1 + x2.T @ reward2 + x3.T @ reward3
        npt.assert_allclose(
            (scorer.avg_A * scorer.sum_weight).numpy(),
            A * scorer.gamma,
            atol=1e-5,
            rtol=1e-5,
        )
        npt.assert_allclose(
            (scorer.avg_b * scorer.sum_weight).numpy(),
            b * scorer.gamma,
            atol=1e-5,
            rtol=1e-5,
        )

        inv_A = np.linalg.inv(A + np.identity(self.x_dim) * scorer.l2_reg_lambda)
        npt.assert_allclose(
            (scorer.inv_avg_A / scorer.sum_weight * scorer.gamma).numpy(),
            inv_A,
            atol=1e-4,
            rtol=1e-4,
        )

        # model parameters are the same
        theta = inv_A @ b
        npt.assert_allclose(scorer.coefs.numpy(), theta, atol=1e-4, rtol=1e-4)

        # ucb scores are the same
        def calculated_expected_ucb_scores(inp):
            expected_out = np.zeros(inp.size()[0])
            for i in range(inp.size()[0]):
                x = inp[i].numpy()
                expected_out[i] = x @ theta.T + scorer.ucb_alpha * np.sqrt(
                    x @ inv_A @ x.T / scorer.gamma
                )
            return expected_out

        expected_out1 = calculated_expected_ucb_scores(inp1)
        npt.assert_allclose(out1.numpy(), expected_out1, atol=1e-4, rtol=1e-4)

        expected_out2 = calculated_expected_ucb_scores(inp2)
        npt.assert_allclose(out2.numpy(), expected_out2, atol=1e-4, rtol=1e-4)
