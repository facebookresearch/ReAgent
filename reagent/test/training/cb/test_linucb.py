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
from reagent.training.cb.linucb_trainer import _get_chosen_arm_features, LinUCBTrainer
from reagent.training.parameters import LinUCBTrainerParameters


class TestLinUCButils(unittest.TestCase):
    def test_get_chosen_arm_features(self):
        all_arms_features = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float
        )
        actions = torch.tensor([[1], [0]], dtype=torch.long)
        chosen_arm_features = _get_chosen_arm_features(all_arms_features, actions)
        npt.assert_equal(
            chosen_arm_features.numpy(), np.array([[3.0, 4.0], [5.0, 6.0]])
        )


class TestLinUCB(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.state_dim = 2
        self.arm_dim = 2

        self.num_arms = 2
        self.params = LinUCBTrainerParameters()

        self.x_dim = 1 + self.state_dim * self.num_arms + self.state_dim + self.num_arms
        policy_network = LinearRegressionUCB(self.x_dim)
        self.policy = Policy(scorer=policy_network, sampler=GreedyActionSampler())

        self.trainer = LinUCBTrainer(self.policy, **self.params.asdict())
        self.batch = CBInput(
            context_arm_features=torch.tensor(
                [
                    [
                        [1, 2, 3, 6, 7, 2 * 6, 2 * 7, 3 * 6, 3 * 7],
                        [1, 2, 3, 10, 11, 2 * 10, 2 * 11, 3 * 10, 3 * 11],
                    ],
                    [
                        [1, 4, 5, 8, 9, 4 * 8, 4 * 9, 5 * 8, 5 * 9],
                        [1, 4, 5, 12, 13, 4 * 12, 4 * 13, 5 * 12, 5 * 13],
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
            np.zeros(scorer_1.A.shape), scorer_1.A.numpy()
        )  # make sure A got updated
        npt.assert_allclose(scorer_1.A.numpy(), scorer_2.A.numpy(), rtol=1e-4)
        npt.assert_allclose(scorer_1.b.numpy(), scorer_2.b.numpy(), rtol=1e-4)

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
            np.zeros(scorer_1.A.shape), scorer_1.A.numpy()
        )  # make sure A got updated
        npt.assert_allclose(scorer_1.A.numpy(), scorer_2.A.numpy(), rtol=1e-4)
        npt.assert_allclose(scorer_1.b.numpy(), scorer_2.b.numpy(), rtol=1e-4)
        npt.assert_allclose(scorer_1.inv_A.numpy(), scorer_2.inv_A.numpy(), rtol=1e-4)
        npt.assert_allclose(scorer_1.coefs.numpy(), scorer_2.coefs.numpy(), rtol=1e-3)

    def test_linucb_model_update_equations(self):
        # make sure that the model parameters match hand-computed values
        scorer = LinearRegressionUCB(self.x_dim)
        policy = Policy(scorer=scorer, sampler=GreedyActionSampler())
        trainer = LinUCBTrainer(policy)
        trainer.training_step(self.batch, 0)
        trainer.on_train_epoch_end()
        # the feature matrix (computed by hand)
        x = _get_chosen_arm_features(
            self.batch.context_arm_features, self.batch.action
        ).numpy()

        npt.assert_allclose(scorer.A.numpy(), x.T @ x, rtol=1e-4)
        npt.assert_allclose(
            scorer.b.numpy(), x.T @ self.batch.reward.squeeze().numpy(), rtol=1e-4
        )

        scorer._calculate_coefs()
        npt.assert_equal(scorer.A.numpy(), scorer.coefs_valid_for_A.numpy())

        npt.assert_allclose(
            (np.eye(self.x_dim) * scorer.l2_reg_lambda + scorer.A.numpy())
            @ scorer.inv_A.numpy(),
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
            np.zeros(scorer_1.A.shape), scorer_1.A.numpy()
        )  # make sure A got updated
        npt.assert_allclose(scorer_1.A.numpy(), scorer_2.A.numpy(), rtol=1e-4)
        npt.assert_allclose(scorer_1.b.numpy(), scorer_2.b.numpy(), rtol=1e-4)
