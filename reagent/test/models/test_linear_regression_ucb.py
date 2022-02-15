#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import numpy as np
import numpy.testing as npt
import torch
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.discrete_sampler import GreedyActionSampler
from reagent.models.linear_regression import (
    LinearRegressionUCB,
    batch_quadratic_form,
)
from reagent.training.cb.linucb_trainer import LinUCBTrainer


class TestLinearRegressionUCBUtils(unittest.TestCase):
    def test_batch_quadratic_form(self) -> None:
        x = torch.tensor([[1.0, 4.3], [3.2, 9.8]])
        A = torch.tensor([[2.0, 1.0], [2.4, 0.5]])
        batch_result = batch_quadratic_form(x, A)
        loop_result = torch.zeros(2)
        for i in range(2):
            loop_result[i] = x[i].t() @ A @ x[i]
        npt.assert_allclose(batch_result.numpy(), loop_result.numpy())


class TestLinearRegressionUCB(unittest.TestCase):
    def test_call_no_ucb(self) -> None:
        x = torch.tensor([[1.0, 2.0], [1.0, 3.0]])  # y=x+1
        y = torch.tensor([3.0, 4.0])
        model = LinearRegressionUCB(2, predict_ucb=False, l2_reg_lambda=0.0)
        trainer = LinUCBTrainer(Policy(scorer=model, sampler=GreedyActionSampler()))
        trainer.update_params(x, y)

        inp = torch.tensor([[1.0, 5.0], [1.0, 6.0]])
        out = model(inp)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(tuple(out.shape), (2,))
        npt.assert_allclose(out.numpy(), np.array([6.0, 7.0]), rtol=1e-5)

    def test_call_ucb(self) -> None:
        x = torch.tensor([[1.0, 2.0], [1.0, 3.0]])  # y=x+1
        y = torch.tensor([3.0, 4.0])
        model = LinearRegressionUCB(2, predict_ucb=True, l2_reg_lambda=0.0)
        trainer = LinUCBTrainer(Policy(scorer=model, sampler=GreedyActionSampler()))
        trainer.update_params(x, y)

        inp = torch.tensor([[1.0, 5.0], [1.0, 6.0]])
        alpha = 1.5
        out = model(inp, ucb_alpha=alpha)

        expected_out = np.zeros(2)
        expected_out[0] = 6.0 + alpha * np.sqrt(
            inp[0].numpy() @ model.inv_A.numpy() @ inp[0].numpy()
        )
        expected_out[1] = 7.0 + alpha * np.sqrt(
            inp[1].numpy() @ model.inv_A.numpy() @ inp[1].numpy()
        )

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(tuple(out.shape), (2,))
        npt.assert_allclose(out.numpy(), expected_out, rtol=1e-6)
