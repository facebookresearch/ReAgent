# pyre-unsafe
import unittest

# pyre-fixme[21]: Could not find module `numpy.testing`.
import numpy.testing as npt
import torch

import torch.nn as nn
from reagent.models.residual_wrapper import ResidualWrapper


class TestResidualWrapper(unittest.TestCase):
    def test_zero_layer(self):
        # if the wrapped layer outputs zero, the residual block should be an identity mapping
        class ZeroLayer(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros_like(x)

        zero_layer = ZeroLayer()
        wrapped_layer = ResidualWrapper(zero_layer)

        x = torch.randn(2, 3)
        y = wrapped_layer(x)
        npt.assert_array_equal(x.detach().numpy(), y.detach().numpy())

    def test_linear_layer(self):
        # test that if the input and output dimensions are the same, the residual block returns the expected output `x+layer(x)`
        linear_layer = torch.nn.Linear(3, 3)
        wrapped_layer = ResidualWrapper(linear_layer)
        x = torch.randn(2, 3)
        y = wrapped_layer(x)
        self.assertEqual(x.shape, y.shape)
        npt.assert_array_equal(
            (x + linear_layer(x)).detach().numpy(), y.detach().numpy()
        )

    def test_mismatched_dims(self):
        # test that if the input and output dimensions are different, the residual block should throw an error during forward pass
        linear_layer = torch.nn.Linear(3, 5)
        wrapped_layer = ResidualWrapper(linear_layer)
        x = torch.randn(2, 3)
        with self.assertRaises(RuntimeError):
            _ = wrapped_layer(x)
