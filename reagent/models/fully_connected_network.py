#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.init as init
from reagent.models.base import ModelBase


logger = logging.getLogger(__name__)


def gaussian_fill_w_gain(tensor, activation, dim_in, min_std=0.0) -> None:
    """ Gaussian initialization with gain."""
    gain = math.sqrt(2) if (activation == "relu" or activation == "leaky_relu") else 1
    init.normal_(tensor, mean=0, std=max(gain * math.sqrt(1 / dim_in), min_std))


ACTIVATION_MAP = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "linear": nn.Identity,
}


class FullyConnectedNetwork(ModelBase):
    def __init__(
        self,
        layers,
        activations,
        *,
        use_batch_norm=False,
        min_std=0.0,
        dropout_ratio=0.0,
        use_layer_norm=False,
        normalize_output=False,
    ) -> None:
        super().__init__()

        self.input_dim = layers[0]

        modules: List[nn.Module] = []

        assert len(layers) == len(activations) + 1

        for i, ((in_dim, out_dim), activation) in enumerate(
            zip(zip(layers, layers[1:]), activations)
        ):
            # Add BatchNorm1d
            if use_batch_norm:
                modules.append(nn.BatchNorm1d(in_dim))
            # Add Linear
            linear = nn.Linear(in_dim, out_dim)
            gaussian_fill_w_gain(linear.weight, activation, in_dim, min_std=min_std)
            init.constant_(linear.bias, 0)  # type: ignore
            modules.append(linear)
            # Add LayerNorm
            if use_layer_norm and (normalize_output or i < len(activations) - 1):
                modules.append(
                    nn.LayerNorm(out_dim)  # type: ignore
                )
            # Add activation
            if activation in ACTIVATION_MAP:
                modules.append(ACTIVATION_MAP[activation]())
            else:
                # See if it matches any of the nn modules
                modules.append(getattr(nn, activation)())
            # Add Dropout
            if dropout_ratio > 0.0 and (normalize_output or i < len(activations) - 1):
                modules.append(nn.Dropout(p=dropout_ratio))

        self.dnn = nn.Sequential(*modules)  # type: ignore

    def input_prototype(self):
        return torch.randn(1, self.input_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass for generic feed-forward DNNs. Assumes activation names
        are valid pytorch activation names.
        :param input tensor
        """
        return self.dnn(input)
