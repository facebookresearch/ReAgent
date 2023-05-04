import unittest

import torch

from reagent.models.deep_represent_linucb import DeepRepresentLinearRegressionUCB


class TestDeepRepresentLinearRegressionUCB(unittest.TestCase):
    """
    This tests the model (not trainer) of DeepRepresentLinUCB.
    """

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

        batch_size = 2
        raw_input = torch.randn(batch_size, input_dim)
        self.assertEqual(
            (batch_size, linucb_inp_dim),  # check deep_represent output size
            model.deep_represent_layers(raw_input).shape,
        )
        model.eval()
        model_output = model(raw_input)

        # check that model's all outputs can be computed
        pred_label = model_output["pred_label"]  # noqa
        pred_sigma = model_output["pred_sigma"]  # noqa
        ucb = model_output["ucb"]
        mlp_out_with_ones = model_output["mlp_out_with_ones"]  # noqa

        self.assertEqual(
            (batch_size,), ucb.shape
        )  # ucb is 1-d tensor of same batch size as input

        self.assertEqual(
            (batch_size,), pred_sigma.shape
        )  # pred_sigma is 1-d tensor of same batch size as input

        self.assertEqual(
            (batch_size,), pred_label.shape
        )  # pred_label is 1-d tensor of same batch size as input
