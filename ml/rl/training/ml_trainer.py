#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np
import scipy
import scipy.stats
import six.moves
from typing import List

# @build:deps [
# @/caffe2/caffe2/python:caffe2_py
# ]

from caffe2.python import workspace, brew
from caffe2.python.model_helper import ModelHelper

from ml.rl.custom_brew_helpers.fc import fc_explicit_param_names
from ml.rl.thrift.core.ttypes import TrainingParameters

brew.Register(fc_explicit_param_names)  # type: ignore


def MakeForwardPassOps(
    model: ModelHelper,
    model_id: str,
    input_blob: str,
    output_blob: str,
    weights: List[str],
    biases: List[str],
    activations: List[str],
    layers: List[int],
    dropout_ratio: float,
    is_test: bool = False,
) -> None:
    """
    Performs a forward pass of a multi-layer perceptron.

    :param model: The ModelHelper object whose net will execute this pass
    :param model_id: A unique string for this model that is used to hold
        activation levels
    :param input_blob: The blob containing the input data
    :param output_blob: The blob where the output data will be placed
    :param weights: A list of blobs containing the weights
    :param biases: A list of blobs containing the bias nodes
    :param activations: A list of strings describing the activation functions
         Currently only 'linear' and 'relu' are supported
    :param layers: A list of integers describing the layer sizes
    :param dropout_ratio: The fraction of nodes to drop out during training.
    :param is_test: Indicates whether or not this forward pass should skip
        node dropout.
    """
    model.net.NanCheck([input_blob], [input_blob])
    num_layer_connections = len(layers) - 1
    for x in six.moves.range(num_layer_connections):
        inputs = None
        outputs = None
        if x == 0:
            inputs = input_blob
        else:
            inputs = "ModelState_" + str(x) + "_" + model_id
        if x + 1 == num_layer_connections:
            outputs = output_blob
        else:
            outputs = "ModelState_" + str(x + 1) + "_" + model_id

        activation = activations[x]
        dim_in = layers[x]
        dim_out = layers[x + 1]
        weight_name = weights[x]
        bias_name = biases[x]

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

        if dropout_ratio > 0.01:
            brew.dropout(
                model, outputs, outputs, ratio=dropout_ratio, is_test=is_test
            )

    model.net.NanCheck([output_blob], [output_blob])


def GenerateLossOps(
    model: ModelHelper,
    output_blob: str,
    label_blob: str,
) -> str:
    """
    Adds loss operators to net. The loss function is computed by a squared L2
    distance, and then averaged over all items in the minibatch.

    :param model: ModelHelper object to add loss operators to.
    :param model_id: String identifier.
    :param output_blob: Blob containing output of net.
    :param label_blob: Blob containing labels.
    :param loss_blob: Blob in which to store loss.
    """
    dist = model.SquaredL2Distance(
        [label_blob, output_blob], model.net.NextBlob("dist")
    )
    loss = model.net.NextBlob('loss')
    model.AveragedLoss(dist, loss)
    return loss


class MLTrainer:
    """ This is meant to be a generic neural net trainer.  It uses minibatch and
    ADAM for momentum/smoothing.
    """

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
        self.model_id = name
        self.optimizer = parameters.optimizer
        self.layers = parameters.layers
        self.activations = parameters.activations
        self.learning_rate = parameters.learning_rate

        self.gamma = parameters.gamma
        self.lr_policy = parameters.lr_policy
        self.dropout_ratio = parameters.dropout_ratio

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

        for x in six.moves.range(len(self.layers) - 1):
            dim_in = self.layers[x]
            dim_out = self.layers[x + 1]

            weight_name = "Weights_" + str(x) + "_" + self.model_id
            bias_name = "Biases_" + str(x) + "_" + self.model_id
            self.weights.append(weight_name)
            self.biases.append(bias_name)

            bias = np.zeros(
                shape=[
                    dim_out,
                ], dtype=np.float32
            )
            workspace.FeedBlob(bias_name, bias)

            gain = np.sqrt(2) if self.activations[x] == 'relu' else 1
            weights = scipy.stats.norm(0, gain * np.sqrt(1 / dim_in)).rvs(
                size=[dim_out, dim_in]
            ).astype(np.float32)
            workspace.FeedBlob(weight_name, weights)

    def build_predictor(self, model, input_blob, output_blob) -> List[str]:
        MakeForwardPassOps(
            model,
            "predictor",
            input_blob,
            output_blob,
            self.weights,
            self.biases,
            self.activations,
            self.layers,
            self.dropout_ratio,
            is_test=True
        )
        return self.weights + self.biases
