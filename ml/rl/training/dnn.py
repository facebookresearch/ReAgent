#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import math
import numpy as np
from typing import List

from caffe2.python import core, workspace, brew
from caffe2.python.model_helper import ModelHelper

from ml.rl.custom_brew_helpers.fc import fc_explicit_param_names
from ml.rl.thrift.core.ttypes import TrainingParameters


class DNN(object):
    """ This class handles evaluating a DNN.  It supports the ml_trainer
        (for inference) and the target_network classes
    """
    registered = False

    def __init__(
        self,
        name: str,
        parameters: TrainingParameters,
    ) -> None:
        """

        :param name: A unique name for this trainer used to create the data on the
            caffe2 workspace
        :param parameters: The set of training parameters
        """
        if not DNN.registered:
            brew.Register(fc_explicit_param_names)  # type: ignore
            DNN.registered = True

        self.model_id = name
        self.layers = parameters.layers
        self.activations = parameters.activations
        self.dropout_ratio = parameters.dropout_ratio
        self.skip_random_weight_init = \
            (parameters.warm_start_model_path is not None)

        self._validate_inputs()
        self._setup_initial_blobs()

    def _validate_inputs(self):
        num_layers = len(self.layers)
        num_activations = len(self.activations)

        if num_activations != num_layers - 1:
            raise Exception(
                "Incompatible input `layers` and `activations` sizes."
            )

        if not all(x > 0 and int(x) == x for x in self.layers):
            raise Exception(
                "All values in `layers` should be positive integers."
            )

    def _setup_initial_blobs(self):
        # Create blobs for model parameters
        self.weights: List[str] = []
        self.biases: List[str] = []

        for x in range(len(self.layers) - 1):
            dim_in = self.layers[x]
            dim_out = self.layers[x + 1]

            weight_name = "Weights_" + str(x) + "_" + self.model_id
            bias_name = "Biases_" + str(x) + "_" + self.model_id
            self.weights.append(weight_name)
            self.biases.append(bias_name)

            if not self.skip_random_weight_init:
                bias = np.zeros(
                    shape=[
                        dim_out,
                    ], dtype=np.float32
                )
                workspace.FeedBlob(bias_name, bias)

                gain = math.sqrt(2) if self.activations[x] == 'relu' else 1
                workspace.RunOperatorOnce(
                    core.CreateOperator(
                        "GaussianFill", [], [weight_name],
                        shape=[dim_out, dim_in],
                        std=gain * math.sqrt(1 / dim_in)
                    )
                )

    def make_forward_pass_ops(
        self,
        model: ModelHelper,
        input_blob: str,
        output_blob: str,
        is_test: bool = False,
    ) -> None:
        """
        Performs a forward pass of a multi-layer perceptron.

        :param model: The ModelHelper object whose net will execute this pass
        :param input_blob: The blob containing the input data
        :param output_blob: The blob where the output data will be placed
        :param is_test: Indicates whether or not this forward pass should skip
            node dropout.
        """
        model.net.NanCheck([input_blob], [input_blob])
        model_states = []
        num_layer_connections = len(self.layers) - 1
        for x in range(num_layer_connections + 1):
            if x == 0:
                model_states.append(input_blob)
            elif x == num_layer_connections:
                model_states.append(output_blob)
            else:
                model_states.append(
                    model.net.
                    NextBlob("ModelState_" + str(x) + "_" + self.model_id)
                )
        for x in range(num_layer_connections):
            inputs = model_states[x]
            outputs = model_states[x + 1]

            activation = self.activations[x]
            dim_in = self.layers[x]
            dim_out = self.layers[x + 1]
            weight_name = self.weights[x]
            bias_name = self.biases[x]

            brew.fc_explicit_param_names(  # type: ignore
                model,
                inputs,
                outputs,
                dim_in=dim_in,
                dim_out=dim_out,
                bias_name=bias_name,
                weight_name=weight_name,
                weight_init=(
                    "GivenTensorFill", {
                        'values': workspace.FetchBlob(weight_name)
                    }
                ),
                bias_init=(
                    "GivenTensorFill", {
                        'values': workspace.FetchBlob(bias_name)
                    }
                )
            )

            if activation == 'relu':
                brew.relu(model, outputs, outputs)
            elif activation == 'linear':
                pass
            else:
                raise Exception("Unknown activation function")

            if self.dropout_ratio > 0.01:
                brew.dropout(
                    model,
                    outputs,
                    outputs,
                    ratio=self.dropout_ratio,
                    is_test=is_test
                )

        model.net.NanCheck([output_blob], [output_blob])
