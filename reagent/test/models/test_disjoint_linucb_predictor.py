#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import unittest

import numpy as np

import numpy.testing as npt
import torch
from reagent.core.types import CBInput
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.discrete_sampler import GreedyActionSampler
from reagent.models.disjoint_linucb_predictor import (
    batch_quadratic_form_multi_arms,
    DisjointLinearRegressionUCB,
)
from reagent.training.cb.disjoint_linucb_trainer import DisjointLinUCBTrainer


class TestDisjointLinearRegressionUCBUtils(unittest.TestCase):
    def test_batch_quadratic_form(self) -> None:
        def check_correctness(x, A):
            batch_result = batch_quadratic_form_multi_arms(x, A)
            batch_size = x.size()[0]
            num_arms = A.size()[0]
            loop_result = torch.zeros((batch_size, num_arms))  # batch_size * num_arms

            for i in range(num_arms):
                a = A[i]
                for j in range(batch_size):
                    loop_result[j][i] = x[j].t() @ a @ x[j]
            npt.assert_allclose(batch_result.numpy(), loop_result.numpy(), rtol=1e-3)

        x1 = torch.tensor([[1.0, 4.3], [3.2, 9.8]])
        A1 = torch.tensor(
            [[[2.0, 1.0], [2.4, 0.5]], [[2.0, 0], [0, 3.0]], [[4.6, 8.0], [3.6, 0.7]]]
        )
        check_correctness(x1, A1)

        torch.manual_seed(0)
        # x (B,N); A (num_arms, N, N)

        # equal num_arms and batch size
        x2 = torch.randn((8, 10))
        A2 = torch.randn((8, 10, 10))
        check_correctness(x2, A2)

        # equal batch size and N
        x3 = torch.randn((8, 8))
        A3 = torch.randn((3, 8, 8))
        check_correctness(x3, A3)

        # equal num_arms and N
        x4 = torch.randn((10, 3))
        A4 = torch.randn((3, 3, 3))
        check_correctness(x4, A4)

        # batch size != N != num_arms
        x5 = torch.randn((10, 8))
        A5 = torch.randn((4, 8, 8))
        check_correctness(x5, A5)

        # batch size = N = num_arms
        x6 = torch.randn((10, 10))
        A6 = torch.randn((10, 10, 10))
        check_correctness(x6, A6)


class TestDisjointLinearRegressionUCB(unittest.TestCase):
    def test_call_ucb(self) -> None:
        inputs = []
        # y = x1+x2
        inputs.append(
            CBInput(
                context_arm_features=torch.tensor([[1.0, 2.0], [1.0, 3.0]]),
                reward=torch.tensor([[3.0], [4.0]]),
            )
        )
        # y = 2x1 + x2
        inputs.append(
            CBInput(
                context_arm_features=torch.tensor([[2.0, 3.0], [1.0, 5.0]]),
                reward=torch.tensor([[7.0], [7.0]]),
            )
        )
        # y = 2x1 + 2x2
        inputs.append(
            CBInput(
                context_arm_features=torch.tensor([[0.5, 3.0], [1.8, 5.1]]),
                reward=torch.tensor([[7.0], [13.8]]),
            )
        )
        model = DisjointLinearRegressionUCB(num_arms=3, input_dim=2, l2_reg_lambda=0.0)
        trainer = DisjointLinUCBTrainer(
            Policy(scorer=model, sampler=GreedyActionSampler())
        )
        trainer.training_step(inputs, batch_idx=0)
        trainer.on_train_epoch_end()

        inp = torch.tensor([[1.0, 5.0], [1.0, 6.0]])
        alpha = 1.5
        out = model(inp, ucb_alpha=alpha)

        def calculate_mean(x, model_type):
            if model_type == 1:
                return x[0] + x[1]
            elif model_type == 2:
                return 2 * x[0] + x[1]
            elif model_type == 3:
                return 2 * x[0] + 2 * x[1]

        expected_out = np.zeros((2, 3))
        for i in range(2):
            x = inp[i]
            for j in range(3):
                expected_out[i, j] = calculate_mean(x, j + 1) + alpha * np.sqrt(
                    x.numpy() @ model.inv_A[j].numpy() @ x.numpy()
                )
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(tuple(out.shape), (2, 3))
        npt.assert_allclose(out.numpy(), expected_out, rtol=1e-4)
