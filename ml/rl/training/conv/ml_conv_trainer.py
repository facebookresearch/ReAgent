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

from ml.rl.custom_brew_helpers.conv import conv_explicit_param_names
from ml.rl.thrift.core.ttypes import TrainingParameters, CNNModelParameters
from ml.rl.training.ml_trainer import MakeForwardPassOps, MLTrainer

brew.Register(conv_explicit_param_names)  # type: ignore


def MakeConvPassOps(
    model: ModelHelper,
    model_id: str,
    dims: List[int],
    conv_height_kernels: List[int],
    conv_width_kernels: List[int],
    pool_kernels_strides: List[int],
    pool_types: List[str],
    input_blob: str,
    output_blob: str,
    conv_weights: List[str],
    conv_biases: List[str]
):
    conv_input_template = "{}_pool_{}"
    conv_output_template = "{}_conv_{}"
    model.net.NanCheck([input_blob], [conv_input_template.format(model_id, 0)])

    for x in six.moves.range(len(dims) - 1):
        conv_input = conv_input_template.format(model_id, x)
        conv_output = conv_output_template.format(model_id, x)
        pool_output = conv_input_template.format(model_id, x + 1)

        pool_type = pool_types[x]
        dim_in = dims[x]
        dim_out = dims[x + 1]
        conv_height_kernel = conv_height_kernels[x]
        conv_width_kernel = conv_width_kernels[x]
        pool_kernel_stride = pool_kernels_strides[x]
        conv_weight_name = conv_weights[x]
        conv_bias_name = conv_biases[x]

        brew.conv_explicit_param_names(  # type: ignore
            model,
            conv_input,
            conv_output,
            dim_in=dim_in,
            dim_out=dim_out,
            kernel_h=conv_height_kernel,
            kernel_w=conv_width_kernel,
            bias_name=conv_bias_name,
            weight_name=conv_weight_name,
            weight_init=(
                "GivenTensorFill", {
                    'values': workspace.FetchBlob(conv_weight_name)
                }
            ),
            bias_init=(
                "GivenTensorFill", {
                    'values': workspace.FetchBlob(conv_bias_name)
                }
            )
        )

        pool_fn = brew.max_pool if pool_type == 'max' else brew.average_pool
        pool_fn(
            model,
            conv_output,
            pool_output,
            kernel=pool_kernel_stride,
            stride=pool_kernel_stride
        )

    model.net.NanCheck([pool_output], [output_blob])


class MLConvTrainer(MLTrainer):
    def __init__(
        self,
        name: str,
        fc_parameters: TrainingParameters,
        cnn_parameters: CNNModelParameters,
        img_height: int,
        img_width: int
    ) -> None:
        self.init_height = img_height
        self.init_width = img_width
        self.dims = cnn_parameters.conv_dims
        self.conv_height_kernels = cnn_parameters.conv_height_kernels
        self.conv_width_kernels = cnn_parameters.conv_width_kernels
        self.pool_kernels_strides = cnn_parameters.pool_kernels_strides
        self.pool_types = cnn_parameters.pool_types

        MLTrainer.__init__(self, name, fc_parameters)

    def _set_conv_dimensions(self):
        heights = [self.init_height]
        widths = [self.init_width]

        for i in range(len(self.conv_height_kernels)):
            heights.append(
                int(
                    (heights[i] - self.conv_height_kernels[i] + 1)
                    / self.pool_kernels_strides[i]
                )
            )
            widths.append(
                int(
                    (widths[i] - self.conv_width_kernels[i] + 1)
                    / self.pool_kernels_strides[i]
                )
            )

        self.layers[0] = self.dims[-1] * heights[-1] * widths[-1]

    def _validate_inputs(self):
        conv_dim_len = len(self.dims) - 1
        if (
            conv_dim_len != len(self.conv_height_kernels)
            or conv_dim_len != len(self.conv_width_kernels)
            or conv_dim_len != len(self.pool_kernels_strides)
            or conv_dim_len != len(self.pool_types)
        ):
            raise Exception(
                "Ensure that `conv_dims`, `conv_height_kernels`, `conv_width_kernels`"
                + ", `pool_kernels`, and `pool_types` are the same length."
            )

        for pool_type in self.pool_types:
            if pool_type not in ['max', 'avg']:
                raise Exception("Unsupported pool type: {}".format(pool_type))

        self._set_conv_dimensions()
        MLTrainer._validate_inputs(self)

    def _setup_initial_blobs(self):
        MLTrainer._setup_initial_blobs(self)

        self.output_conv_blob = "Conv_output_{}".format(self.model_id)
        workspace.FeedBlob(self.output_conv_blob, np.zeros(1, dtype=np.float32))

        self.conv_weights: List[str] = []
        self.conv_biases: List[str] = []

        for x in six.moves.range(len(self.dims) - 1):
            dim_in = self.dims[x]
            dim_out = self.dims[x + 1]
            kernel_h = self.conv_height_kernels[x]
            kernel_w = self.conv_width_kernels[x]

            weight_shape = [dim_out, kernel_h, kernel_w, dim_in]
            bias_shape = [dim_out, ]

            conv_weight_name = "ConvWeights_" + str(x) + "_" + self.model_id
            bias_name = "ConvBiases_" + str(x) + "_" + self.model_id
            self.conv_weights.append(conv_weight_name)
            self.conv_biases.append(bias_name)

            conv_bias = np.zeros(shape=bias_shape, dtype=np.float32)
            workspace.FeedBlob(bias_name, conv_bias)

            conv_weights = scipy.stats.norm(0, np.sqrt(1 / dim_in)).rvs(
                size=weight_shape
            ).astype(np.float32)
            workspace.FeedBlob(conv_weight_name, conv_weights)

    def _build_fwd_pass_train_model(self):
        self.train_model.StopGradient(self.labels_blob, self.labels_blob)
        model_id = self.model_id + "_train"
        MakeConvPassOps(
            self.train_model, model_id, self.dims, self.conv_height_kernels,
            self.conv_width_kernels, self.pool_kernels_strides, self.pool_types,
            self.input_blob, self.output_conv_blob, self.conv_weights,
            self.conv_biases
        )
        MakeForwardPassOps(
            self.train_model, model_id, self.output_conv_blob,
            self.output_blob, self.weights, self.biases, self.activations,
            self.layers, self.dropout_ratio, False
        )

    def _build_fwd_pass_score_model(self):
        model_id = self.model_id + "_score"
        MakeConvPassOps(
            self.score_model, model_id, self.dims, self.conv_height_kernels,
            self.conv_width_kernels, self.pool_kernels_strides, self.pool_types,
            self.input_blob, self.output_conv_blob, self.conv_weights,
            self.conv_biases
        )
        MakeForwardPassOps(
            self.score_model, model_id, self.output_conv_blob, self.output_blob,
            self.weights, self.biases, self.activations, self.layers,
            self.dropout_ratio, False
        )
