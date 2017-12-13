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
from ml.rl.training.model_update_helper import AddParameterUpdateOps

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
    model: ModelHelper, model_id: str, output_blob: str, label_blob: str,
    loss_blob: str
) -> None:
    """
    Adds loss operators to net. The loss function is computed by a squared L2
    distance, and then averaged over all items in the minibatch.

    :param model: ModelHelper object to add loss operators to.
    :param model_id: String identifier.
    :param output_blob: Blob containing output of net.
    :param label_blob: Blob containing labels.
    :param loss_blob: Blob in which to store loss.
    """
    dist = model.SquaredL2Distance([label_blob, output_blob], model_id + "dist")
    model.AveragedLoss(dist, loss_blob)


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

        self._build_fwd_pass_score_model()

        self._build_fwd_pass_train_model()
        self._generate_train_model_loss()
        self._update_train_model()

        workspace.RunNetOnce(self.score_model.param_init_net)
        workspace.CreateNet(self.score_model.net)
        workspace.RunNetOnce(self.train_model.param_init_net)
        workspace.CreateNet(self.train_model.net)

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

    def _generate_train_model_loss(self):
        GenerateLossOps(
            self.train_model, self.model_id + "_train", self.output_blob,
            self.labels_blob, self.loss_blob
        )

    def _build_fwd_pass_train_model(self):
        self.train_model.StopGradient(self.labels_blob, self.labels_blob)
        MakeForwardPassOps(
            self.train_model, self.model_id + "_train", self.input_blob,
            self.output_blob, self.weights, self.biases, self.activations,
            self.layers, self.dropout_ratio, False
        )

    def _build_fwd_pass_score_model(self):
        MakeForwardPassOps(
            self.score_model, self.model_id + "_score", self.input_blob,
            self.output_blob, self.weights, self.biases, self.activations,
            self.layers, self.dropout_ratio, True
        )

    def _setup_initial_blobs(self):
        # Define models
        self.score_model = ModelHelper(name="score_" + self.model_id)
        self.train_model = ModelHelper(name="train_" + self.model_id)

        # Create input, output, labels, and loss blobs
        self.input_blob = "ModelInput_" + self.model_id
        workspace.FeedBlob(self.input_blob, np.zeros(1, dtype=np.float32))
        self.output_blob = "ModelOutput_" + self.model_id
        workspace.FeedBlob(self.output_blob, np.zeros(1, dtype=np.float32))
        self.labels_blob = "ModelLabels_" + self.model_id
        workspace.FeedBlob(self.labels_blob, np.zeros(1, dtype=np.float32))
        self.loss_blob = "loss"  # "ModelLoss_" + self.model_id
        workspace.FeedBlob(self.loss_blob, np.zeros(1, dtype=np.float32))

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

    def _update_train_model(self):
        self.train_model.AddGradientOperators([self.loss_blob])

        for param in self.train_model.params:
            param_grad = self.train_model.param_to_grad[param]
            self.train_model.net.NanCheck([param_grad], [param_grad])

        AddParameterUpdateOps(
            self.train_model,
            optimizer_input=self.optimizer,
            base_learning_rate=self.learning_rate,
            gamma=self.gamma,
            policy=self.lr_policy,
        )

    def build_predictor(self, model, input_blob, output_blob) -> List[str]:
        MakeForwardPassOps(
            model,
            self.model_id + "_score",
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

    def score(self, inputs: np.ndarray) -> np.ndarray:
        """
        Runs the net on a set of data and returns the outputs.

        :param inputs: Numpy array containing examples to score.
        """
        workspace.FeedBlob(self.input_blob, inputs)
        workspace.RunNet(self.score_model.net)
        return workspace.FetchBlob(self.output_blob)

    def train_batch(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        """
        Trains net on inputs and labels. Please ensure that inputs are batched
        to an appropriate size and are shuffled.

        :param inputs: Numpy array containing training examples.
        :param labels: Numpy array containing training labels.
        """
        workspace.FeedBlob(self.input_blob, inputs)
        workspace.FeedBlob(self.labels_blob, labels)
        workspace.RunNet(self.train_model.net)

    @property
    def output(self) -> np.ndarray:
        return workspace.FetchBlob(self.output_blob)

    @property
    def loss(self) -> np.ndarray:
        return workspace.FetchBlob('loss')
