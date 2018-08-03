#!/usr/bin/env python3

import unittest

import numpy as np
from caffe2.python import brew, model_helper, workspace
from ml.rl.custom_brew_helpers.fc import fc_explicit_param_names
from scipy import stats


class TestCustomBrewHelpers(unittest.TestCase):
    def fc_explicit_param_names(self):
        brew.Register(fc_explicit_param_names)
        model = model_helper.ModelHelper(name="test_model")
        dim_in = 10
        dim_out = 100
        weight_name = "test_weight_name"
        bias_name = "test_bias_name"
        inputs_name = "test_inputs"
        output_name = "test_output"

        input_distribution = stats.norm()
        inputs = input_distribution.rvs(size=(1, dim_in)).astype(np.float32)
        workspace.FeedBlob(inputs_name, inputs)

        weights = np.random.normal(size=(dim_out, dim_in)).astype(np.float32)
        bias = np.transpose(np.random.normal(size=(dim_out)).astype(np.float32))

        brew.fc_explicit_param_names(
            model,
            inputs_name,
            output_name,
            dim_in=dim_in,
            dim_out=dim_out,
            bias_name=bias_name,
            weight_name=weight_name,
            weight_init=("GivenTensorFill", {"values": weights}),
            bias_init=("GivenTensorFill", {"values": bias}),
        )

        workspace.RunNetOnce(model.param_init_net)

        model.net.Proto().type = "async_scheduling"
        workspace.CreateNet(model.net)

        workspace.RunNet(model.net)

        expected_output = np.dot(inputs, np.transpose(weights)) + bias
        outputs_diff = expected_output - workspace.FetchBlob(output_name)

        self.assertEqual(np.linalg.norm(outputs_diff), 0)
