import unittest

import torch

from reagent.models.deep_represent_linucb import DeepRepresentLinearRegressionUCB


class TestDeepRepresentLinearRegressionUCB(unittest.TestCase):
    def test_basic(self):
        input_dim = 9
        sizes = [6]
        linucb_inp_dim = 3
        activations = ["relu", "relu"]

        model = DeepRepresentLinearRegressionUCB(
            input_dim=input_dim,
            sizes=sizes + [linucb_inp_dim],
            activations=activations,
        )

        batch_size = 3
        raw_input = torch.randn(batch_size, input_dim)
        self.assertEqual(
            (batch_size, linucb_inp_dim),  # check deep_represent output size
            model.deep_represent_layers(raw_input).shape,
        )
        model.eval()
        output = model(raw_input)  # check final output size
        self.assertEqual(
            (batch_size,), output.shape
        )  # ucb is 1-d tensor of same batch size as input
