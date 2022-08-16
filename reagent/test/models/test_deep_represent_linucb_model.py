import unittest

from reagent.models.deep_represent_linucb import DeepRepresentLinearRegressionUCB


class TestDeepRepresentLinearRegressionUCB(unittest.TestCase):
    def test_basic(self):
        raw_input_dim = 9
        sizes = [6]
        linucb_inp_dim = 3
        activations = ["relu"]

        model = DeepRepresentLinearRegressionUCB(
            raw_input_dim=raw_input_dim,
            sizes=sizes,
            linucb_inp_dim=linucb_inp_dim,
            activations=activations,
        )

        raw_input = model.input_prototype()
        self.assertEqual((1, raw_input_dim), raw_input.shape)  # check input size
        self.assertEqual(
            (1, linucb_inp_dim),  # check deep_represent output size
            model.deep_represent_layers(raw_input).shape,
        )
        model.eval()
        output = model(raw_input)  # check final output size
        self.assertEqual((1,), output.shape)  # ucb is 0-d tensor
