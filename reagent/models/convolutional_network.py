#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import collections
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from reagent.models.fully_connected_network import FullyConnectedNetwork


logger = logging.getLogger(__name__)


CnnParameters = collections.namedtuple(
    "CnnParameters",
    [
        "conv_dims",
        "conv_height_kernels",
        "conv_width_kernels",
        "pool_types",
        "pool_kernels_strides",
        "num_input_channels",
        "input_height",
        "input_width",
    ],
)


class ConvolutionalNetwork(nn.Module):
    def __init__(self, cnn_parameters, layers, activations, use_layer_norm) -> None:
        super().__init__()
        self.conv_dims = cnn_parameters.conv_dims
        self.conv_height_kernels = cnn_parameters.conv_height_kernels
        self.conv_width_kernels = cnn_parameters.conv_width_kernels
        self.use_layer_norm = use_layer_norm

        self.conv_layers: nn.ModuleList = nn.ModuleList()
        self.pool_layers: nn.ModuleList = nn.ModuleList()
        self.layer_norm_layers: nn.ModuleList = nn.ModuleList()

        for i, _ in enumerate(self.conv_dims[1:]):
            self.conv_layers.append(
                nn.Conv2d(
                    self.conv_dims[i],
                    self.conv_dims[i + 1],
                    kernel_size=(
                        self.conv_height_kernels[i],
                        self.conv_width_kernels[i],
                    ),
                )
            )
            nn.init.kaiming_normal_(self.conv_layers[i].weight)
            if cnn_parameters.pool_types[i] == "max":
                self.pool_layers.append(
                    nn.MaxPool2d(kernel_size=cnn_parameters.pool_kernels_strides[i])
                )
            else:
                assert False, "Unknown pooling type".format(layers)
            if self.use_layer_norm:
                self.layer_norm_layers.append(nn.GroupNorm(1, self.conv_dims[i + 1]))

        input_size = (
            cnn_parameters.num_input_channels,
            cnn_parameters.input_height,
            cnn_parameters.input_width,
        )
        conv_out = self.conv_forward(torch.ones(1, *input_size))
        self.fc_input_dim = int(np.prod(conv_out.size()[1:]))
        layers[0] = self.fc_input_dim
        self.feed_forward = FullyConnectedNetwork(
            layers, activations, use_layer_norm=use_layer_norm
        )

    def conv_forward(self, input):
        x = input
        for i, _ in enumerate(self.conv_layers):
            x = self.conv_layers[i](x)
            if self.use_layer_norm:
                x = self.layer_norm_layers[i](x)
            x = F.relu(x)
            x = self.pool_layers[i](x)
        return x

    def forward(self, input) -> torch.FloatTensor:
        """Forward pass for generic convnet DNNs. Assumes activation names
        are valid pytorch activation names.
        :param input image tensor
        """
        x = self.conv_forward(input)
        x = x.view(-1, self.fc_input_dim)
        # pyre-fixme[7]: Expected `FloatTensor` but got `Tensor`.
        return self.feed_forward.forward(x)
