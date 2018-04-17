#!/usr/bin/env python3


import math
import numpy as np
from typing import List, Optional

from caffe2.python import core, workspace, brew
from caffe2.python.model_helper import ModelHelper

from ml.rl.custom_brew_helpers.conv import conv_explicit_param_names
from ml.rl.thrift.core.ttypes import CNNParameters


class CNN(object):
    """ This class handles evaluating a Convolutional Neural Network (CNN).
        It supports the ml_trainer (for inference) and the target_network
        classes
    """
    registered = False

    def __init__(
        self,
        name: str,
        cnn_parameters: CNNParameters,
    ) -> None:
        """

        :param name: A unique name for this trainer used to create the data on the
            caffe2 workspace
        :param parameters: The set of training parameters
        """

        if not CNN.registered:
            brew.Register(conv_explicit_param_names)  # type: ignore
            CNN.registered = True

        self.model_id = name

        self.init_height = cnn_parameters.input_height
        self.init_width = cnn_parameters.input_width
        self.dims = cnn_parameters.conv_dims
        self.conv_height_kernels = cnn_parameters.conv_height_kernels
        self.conv_width_kernels = cnn_parameters.conv_width_kernels
        self.pool_kernels_strides = cnn_parameters.pool_kernels_strides
        self.pool_types = cnn_parameters.pool_types

        self._validate_inputs()
        self._setup_initial_blobs()

    def get_output_size(self):
        heights = [self.init_height]
        widths = [self.init_width]

        for i in range(len(self.conv_height_kernels)):
            heights.append(
                int(
                    (heights[i] - self.conv_height_kernels[i] + 1) /
                    self.pool_kernels_strides[i]
                )
            )
            widths.append(
                int(
                    (widths[i] - self.conv_width_kernels[i] + 1) /
                    self.pool_kernels_strides[i]
                )
            )

        return self.dims[-1] * heights[-1] * widths[-1]

    def _validate_inputs(self):
        conv_dim_len = len(self.dims) - 1
        if (
            conv_dim_len != len(self.conv_height_kernels) or
            conv_dim_len != len(self.conv_width_kernels) or
            conv_dim_len != len(self.pool_kernels_strides) or
            conv_dim_len != len(self.pool_types)
        ):
            raise Exception(
                "Ensure that `conv_dims`, `conv_height_kernels`, `conv_width_kernels`"
                + ", `pool_kernels`, and `pool_types` are the same length."
            )

        for pool_type in self.pool_types:
            if pool_type not in ['max', 'avg']:
                raise Exception("Unsupported pool type: {}".format(pool_type))

    def _setup_initial_blobs(self):
        self.output_conv_blob = "Conv_output_{}".format(self.model_id)
        workspace.FeedBlob(self.output_conv_blob, np.zeros(1, dtype=np.float32))

        self.weights: List[str] = []
        self.biases: List[str] = []

        for x in range(len(self.dims) - 1):
            dim_in = self.dims[x]
            dim_out = self.dims[x + 1]
            kernel_h = self.conv_height_kernels[x]
            kernel_w = self.conv_width_kernels[x]

            weight_shape = [dim_out, kernel_h, kernel_w, dim_in]
            bias_shape = [
                dim_out,
            ]

            conv_weight_name = "ConvWeights_" + str(x) + "_" + self.model_id
            bias_name = "ConvBiases_" + str(x) + "_" + self.model_id
            self.weights.append(conv_weight_name)
            self.biases.append(bias_name)

            conv_bias = np.zeros(shape=bias_shape, dtype=np.float32)
            workspace.FeedBlob(bias_name, conv_bias)

            workspace.RunOperatorOnce(
                core.CreateOperator(
                    "GaussianFill", [], [conv_weight_name],
                    shape=weight_shape,
                    std=math.sqrt(1 / dim_in)
                )
            )

    def make_conv_pass_ops(
        self,
        model: ModelHelper,
        input_blob: str,
        output_blob: str,
    ):
        conv_input_template = "{}_pool_{}"
        conv_output_template = "{}_conv_{}"
        model.net.NanCheck(
            [input_blob], [conv_input_template.format(self.model_id, 0)]
        )

        for x in range(len(self.dims) - 1):
            conv_input = conv_input_template.format(self.model_id, x)

            pool_kernel_stride = self.pool_kernels_strides[x]

            if pool_kernel_stride > 1:
                conv_output = conv_output_template.format(self.model_id, x)
                pool_output: Optional[str] = conv_input_template.format(
                    self.model_id, x + 1
                )
                pool_type: Optional[str] = self.pool_types[x]
            else:
                conv_output = conv_input_template.format(self.model_id, x + 1)
                pool_output = None
                pool_type = None

            dim_in = self.dims[x]
            dim_out = self.dims[x + 1]
            conv_height_kernel = self.conv_height_kernels[x]
            conv_width_kernel = self.conv_width_kernels[x]
            conv_weight_name = self.weights[x]
            conv_bias_name = self.biases[x]

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

            if pool_kernel_stride > 1:
                pool_fn = brew.max_pool if pool_type == 'max' else brew.average_pool
                pool_fn(
                    model,
                    conv_output,
                    pool_output,
                    kernel=pool_kernel_stride,
                    stride=pool_kernel_stride
                )

        if pool_output:
            model.net.NanCheck([pool_output], [output_blob])
        else:
            model.net.NanCheck([conv_output], [output_blob])
