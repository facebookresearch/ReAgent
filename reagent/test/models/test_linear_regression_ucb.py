#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import unittest

import numpy as np

# pyre-fixme[21]: Could not find module `numpy.testing`.
import numpy.testing as npt
import torch
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.discrete_sampler import GreedyActionSampler
from reagent.models.linear_regression import batch_quadratic_form, LinearRegressionUCB
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

    def test_batch_quadratic_form_3d(self) -> None:
        x = torch.tensor([[[1.0, 4.3], [3.2, 9.8]], [[1.2, 4.1], [3.0, 7.8]]])
        A = torch.tensor([[2.0, 1.0], [2.4, 0.5]])
        batch_result = batch_quadratic_form(x, A)
        loop_result = torch.zeros(2, 2)
        for i in range(2):
            for j in range(2):
                loop_result[i][j] = x[i, j].t() @ A @ x[i, j]
        npt.assert_allclose(batch_result.numpy(), loop_result.numpy())


class TestLinearRegressionUCB(unittest.TestCase):
    def test_call_no_ucb(self) -> None:
        x = torch.tensor([[1.0, 2.0], [1.0, 3.0]])  # y=x+1
        y = torch.tensor([3.0, 4.0])
        model = LinearRegressionUCB(2, ucb_alpha=0, l2_reg_lambda=0.0)
        trainer = LinUCBTrainer(Policy(scorer=model, sampler=GreedyActionSampler()))
        trainer.update_params(x, y)

        inp = torch.tensor([[1.0, 5.0], [1.0, 6.0]])
        model_output = model(inp)
        ucb = model_output["ucb"]

        self.assertIsInstance(ucb, torch.Tensor)
        self.assertEqual(tuple(ucb.shape), (2,))
        npt.assert_allclose(ucb.numpy(), np.array([6.0, 7.0]), rtol=1e-4)

    def test_call_ucb(self) -> None:
        x = torch.tensor([[1.0, 2.0], [1.0, 3.0]])  # y=x+1
        y = torch.tensor([3.0, 4.0])
        model = LinearRegressionUCB(2, l2_reg_lambda=0.0)
        trainer = LinUCBTrainer(Policy(scorer=model, sampler=GreedyActionSampler()))
        trainer.update_params(x, y)

        inp = torch.tensor([[1.0, 5.0], [1.0, 6.0]])
        alpha = 1.5
        model_output = model(inp, ucb_alpha=alpha)
        ucb = model_output["ucb"]

        expected_out = np.zeros(2)
        expected_out[0] = 6.0 + alpha * np.sqrt(
            inp[0].numpy()
            @ (model.inv_avg_A / model.sum_weight).numpy()
            @ inp[0].numpy()
        )
        expected_out[1] = 7.0 + alpha * np.sqrt(
            inp[1].numpy()
            @ (model.inv_avg_A / model.sum_weight).numpy()
            @ inp[1].numpy()
        )

        self.assertIsInstance(ucb, torch.Tensor)
        self.assertEqual(tuple(ucb.shape), (2,))
        npt.assert_allclose(ucb.numpy(), expected_out, rtol=1e-4)
