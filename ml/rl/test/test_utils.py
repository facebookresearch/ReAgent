#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import numpy as np
import numpy.testing as npt
import torch
from ml.rl.caffe_utils import masked_softmax
from ml.rl.training.rl_trainer_pytorch import rescale_torch_tensor


class TestUtils(unittest.TestCase):
    def test_rescale_torch_tensor(self):
        rows, cols = 3, 5
        original_tensor = torch.randint(low=10, high=40, size=(rows, cols)).float()
        prev_max_tensor = torch.ones(1, 5) * 40.0
        prev_min_tensor = torch.ones(1, 5) * 10.0
        new_min_tensor = torch.ones(1, 5) * -1.0
        new_max_tensor = torch.ones(1, 5).float()

        print("Original tensor: ", original_tensor)
        rescaled_tensor = rescale_torch_tensor(
            original_tensor,
            new_min_tensor,
            new_max_tensor,
            prev_min_tensor,
            prev_max_tensor,
        )
        print("Rescaled tensor: ", rescaled_tensor)
        reconstructed_original_tensor = rescale_torch_tensor(
            rescaled_tensor,
            prev_min_tensor,
            prev_max_tensor,
            new_min_tensor,
            new_max_tensor,
        )
        print("Reconstructed Original tensor: ", reconstructed_original_tensor)

        comparison_tensor = torch.eq(original_tensor, reconstructed_original_tensor)
        self.assertTrue(torch.sum(comparison_tensor), rows * cols)

    def test_masked_softmax(self):
        # Postive value case
        x = torch.tensor([[15.0, 6.0, 9.0], [3.0, 2.0, 1.0]])
        temperature = 1
        mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        out = masked_softmax(x, mask, temperature)
        expected_out = torch.tensor([[0.9975, 0.0000, 0.0025], [0, 0.7311, 0.2689]])
        npt.assert_array_almost_equal(out, expected_out, 4)

        # Postive value case (masked value goes to inf)
        x = torch.tensor([[150.0, 2.0]])
        temperature = 0.01
        mask = torch.tensor([[0.0, 1.0]])
        out = masked_softmax(x, mask, temperature)
        expected_out = torch.tensor([[0.0, 1.0]])
        npt.assert_array_almost_equal(out, expected_out, 4)

        # Negative value case
        x = torch.tensor([[-10.0, -1.0, -5.0]])
        temperature = 0.01
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        out = masked_softmax(x, mask, temperature)
        expected_out = torch.tensor([[0.0, 1.0, 0.0]])
        npt.assert_array_almost_equal(out, expected_out, 4)

        # All values in a row are masked case
        x = torch.tensor([[-5.0, 4.0, 3.0], [2.0, 1.0, 2.0]])
        temperature = 1
        mask = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        out = masked_softmax(x, mask, temperature)
        expected_out = torch.tensor([[0.0, 0.0, 0.0], [0.4223, 0.1554, 0.4223]])
        npt.assert_array_almost_equal(out, expected_out, 4)
