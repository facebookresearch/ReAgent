#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import unittest

import numpy.testing as npt
import torch
from reagent.core.torch_utils import (
    masked_softmax,
    reorder_data_kjt,
    rescale_torch_tensor,
    shift_kjt_by_one,
    split_sequence_keyed_jagged_tensor,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TestUtils(unittest.TestCase):
    def test_rescale_torch_tensor(self) -> None:
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

    def test_masked_softmax(self) -> None:
        # Postive value case
        x = torch.tensor([[15.0, 6.0, 9.0], [3.0, 2.0, 1.0]])
        temperature = 1
        mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        # pyre-fixme[6]: For 2nd param expected `float` but got `Tensor`.
        out = masked_softmax(x, mask, temperature)
        expected_out = torch.tensor([[0.9975, 0.0000, 0.0025], [0, 0.7311, 0.2689]])
        npt.assert_array_almost_equal(out, expected_out, 4)

        # Postive value case (masked value goes to inf)
        x = torch.tensor([[150.0, 2.0]])
        temperature = 0.01
        mask = torch.tensor([[0.0, 1.0]])
        # pyre-fixme[6]: For 2nd param expected `float` but got `Tensor`.
        out = masked_softmax(x, mask, temperature)
        expected_out = torch.tensor([[0.0, 1.0]])
        npt.assert_array_almost_equal(out, expected_out, 4)

        # Negative value case
        x = torch.tensor([[-10.0, -1.0, -5.0]])
        temperature = 0.01
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        # pyre-fixme[6]: For 2nd param expected `float` but got `Tensor`.
        out = masked_softmax(x, mask, temperature)
        expected_out = torch.tensor([[0.0, 1.0, 0.0]])
        npt.assert_array_almost_equal(out, expected_out, 4)

        # All values in a row are masked case
        x = torch.tensor([[-5.0, 4.0, 3.0], [2.0, 1.0, 2.0]])
        temperature = 1
        mask = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        # pyre-fixme[6]: For 2nd param expected `float` but got `Tensor`.
        out = masked_softmax(x, mask, temperature)
        expected_out = torch.tensor([[0.0, 0.0, 0.0], [0.4223, 0.1554, 0.4223]])
        npt.assert_array_almost_equal(out, expected_out, 4)

    def test_split_sequence_keyed_jagged_tensor(self) -> None:
        """Test the example in the docstring of split_sequence_keyed_jagged_tensor"""
        keys = ["Key0", "Key1", "Key2"]
        values = torch.arange(10).float()
        weights = values / 10.0
        lengths = torch.tensor([2, 0, 1, 1, 1, 1, 3, 0, 0, 1, 0, 0])
        num_steps = 2

        def verify_output(out):
            self.assertEqual(out[0].keys(), keys)
            assert torch.allclose(
                out[0].values(), torch.tensor([0.0, 1.0, 2.0, 4.0, 6.0, 7.0, 8.0])
            )
            assert torch.allclose(out[0].lengths(), torch.tensor([2, 1, 1, 3, 0, 0]))
            if out[0]._weights is not None:
                assert torch.allclose(
                    out[0].weights(), torch.tensor([0.0, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8])
                )
            assert torch.allclose(out[1].values(), torch.tensor([3.0, 5.0, 9.0]))
            assert torch.allclose(out[1].lengths(), torch.tensor([0, 1, 1, 0, 1, 0]))
            if out[1]._weights is not None:
                assert torch.allclose(out[1].weights(), torch.tensor([0.3, 0.5, 0.9]))

        # Test id list data
        x0 = KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)
        y0 = split_sequence_keyed_jagged_tensor(x0, num_steps)
        verify_output(y0)

        # Test id score list data
        x1 = KeyedJaggedTensor(
            keys=keys, values=values, lengths=lengths, weights=weights
        )
        y1 = split_sequence_keyed_jagged_tensor(x1, num_steps)
        verify_output(y1)

    def test_reorder_data_kjt(self) -> None:
        """Test the example in the docstring of reorder_data_kjt"""
        keys = ["Key0", "Key1"]
        values = torch.arange(7).float()
        weights = values / 10.0
        lengths = torch.tensor([2, 0, 1, 1, 1, 2])

        # With weights
        x = KeyedJaggedTensor(
            keys=keys, values=values, lengths=lengths, weights=weights
        )
        y = reorder_data_kjt(x, torch.tensor([2, 1, 0]))
        self.assertEqual(y.keys(), keys)
        assert torch.allclose(
            y.values(), torch.tensor([2.0, 0.0, 1.0, 5.0, 6.0, 4.0, 3.0])
        )
        assert torch.allclose(y.lengths(), torch.tensor([1, 0, 2, 2, 1, 1]))
        assert torch.allclose(y.weights(), y.values() / 10.0)

        # Without weights
        x = KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)
        y = reorder_data_kjt(x, torch.tensor([2, 1, 0]))
        self.assertEqual(y.keys(), keys)
        assert torch.allclose(
            y.values(), torch.tensor([2.0, 0.0, 1.0, 5.0, 6.0, 4.0, 3.0])
        )
        assert torch.allclose(y.lengths(), torch.tensor([1, 0, 2, 2, 1, 1]))

    def test_shift_kjt_by_one(self) -> None:
        """Test the example in the docstring of shift_kjt_by_one"""
        keys = ["Key0", "Key1"]
        values = torch.arange(7).float()
        weights = values / 10.0
        lengths = torch.tensor([2, 0, 1, 1, 1, 2])

        # With weights
        x = KeyedJaggedTensor(
            keys=keys, values=values, lengths=lengths, weights=weights
        )
        y = shift_kjt_by_one(x)
        self.assertEqual(y.keys(), keys)
        assert torch.allclose(y.values(), torch.tensor([2.0, 4.0, 5.0, 6.0]))
        assert torch.allclose(y.lengths(), torch.tensor([0, 1, 0, 1, 2, 0]))
        assert torch.allclose(y.weights(), y.values() / 10.0)

        # Without weights
        x = KeyedJaggedTensor(
            keys=keys,
            values=values,
            lengths=lengths,
        )
        y = shift_kjt_by_one(x)
        self.assertEqual(y.keys(), keys)
        assert torch.allclose(y.values(), torch.tensor([2.0, 4.0, 5.0, 6.0]))
        assert torch.allclose(y.lengths(), torch.tensor([0, 1, 0, 1, 2, 0]))
