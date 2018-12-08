#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import numpy as np
import torch
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
