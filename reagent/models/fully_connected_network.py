#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.init as init
from reagent.core import types as rlt
from reagent.models.base import ModelBase
from reagent.models.residual_wrapper import ResidualWrapper


logger = logging.getLogger(__name__)


def gaussian_fill_w_gain(tensor, gain, dim_in, min_std=0.0) -> None:
    """Gaussian initialization with gain."""
    init.normal_(tensor, mean=0, std=max(gain * math.sqrt(1 / dim_in), min_std))


# troch.fx.trace does not support dynamic control flow, wrap the if-else and assert logic in this function to work around this limitation
@torch.fx.wrap
def transpose_tensor(shape_tensor: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    shape = len(shape_tensor.shape)
    assert shape in [2, 3], f"Invalid input shape {shape}"
    if shape == 2:
        return input
    else:
        return input.transpose(1, 2)


ACTIVATION_MAP = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "linear": nn.Identity,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
}


class SlateBatchNorm1d(nn.Module):
    """
    Same as nn.BatchNorm1d is input has shape (batch_size, feat_dim).
    But if input has shape (batch_size, num_candidates, item_feats), like in LearnedVM,
    we transpose it, since that's what nn.BatchNorm1d computes Batch Normalization over
    1st dimension, while we want to compute it over item_feats.

    NOTE: this is different from nn.BatchNorm2d which is for CNNs, and expects 4D inputs
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.vanilla = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        input = transpose_tensor(x, x)
        output = self.vanilla(input)
        return transpose_tensor(x, output)


class FullyConnectedNetwork(ModelBase):
    def __init__(
        self,
        layers,
        activations,
        *,
        use_batch_norm: bool = False,
        min_std: float = 0.0,
        dropout_ratio: float = 0.0,
        use_layer_norm: bool = False,
        normalize_output: bool = False,
        orthogonal_init: bool = False,
        use_skip_connections: bool = False,
    ) -> None:
        """
        A Fully Connected (FC) network that supports:
        1. Batch Normalization
        2. Dropout
        3. Layer Normalization
        4. Output normalization
        5. Orthogonal initialization
        6. Skip (residual) connections (added only to layers with the same input and output dims)
        """
        super().__init__()

        self.input_dim = layers[0]

        # this list stores all the layers in the network
        modules: List[nn.Module] = []

        assert (
            len(layers) == len(activations) + 1
        ), f"Invalid number of layers {len(layers)} and activations {len(activations)}. Number of layers needs to be 1 + number of activations"

        for i, ((in_dim, out_dim), activation) in enumerate(
            zip(zip(layers, layers[1:]), activations)
        ):
            # build up each layer by stacking the components of the layer
            layer_components: List[nn.Module] = []

            # Add BatchNorm1d
            if use_batch_norm:
                layer_components.append(SlateBatchNorm1d(in_dim))

            # Add Linear
            linear = nn.Linear(in_dim, out_dim)
            # assuming activation is valid
            try:
                gain = torch.nn.init.calculate_gain(activation)
            except ValueError:
                gain = 1.0  # default value for other activation functions
            if orthogonal_init:
                # provably better https://openreview.net/forum?id=rkgqN1SYvr
                nn.init.orthogonal_(linear.weight.data, gain=gain)
            else:
                # gaussian init
                gaussian_fill_w_gain(
                    linear.weight, gain=gain, dim_in=in_dim, min_std=min_std
                )
            init.constant_(linear.bias, 0)  # type: ignore
            layer_components.append(linear)

            # Add LayerNorm
            if use_layer_norm and (normalize_output or i < len(activations) - 1):
                layer_components.append(nn.LayerNorm(out_dim))  # type: ignore

            # Add activation
            if activation in ACTIVATION_MAP:
                layer_components.append(ACTIVATION_MAP[activation]())
            else:
                # See if it matches any of the nn modules
                layer_components.append(getattr(nn, activation)())

            # Add Dropout
            if dropout_ratio > 0.0 and (normalize_output or i < len(activations) - 1):
                layer_components.append(nn.Dropout(p=dropout_ratio))

            layer = nn.Sequential(*layer_components)
            if use_skip_connections:
                if in_dim == out_dim:
                    layer = ResidualWrapper(layer)
                else:
                    logger.warn(
                        f"Skip connections are enabled, but layer in_dim ({in_dim}) != out_dim ({out_dim}). Skip connection will not be added for this layer"
                    )
            modules.append(layer)
        self.dnn = nn.Sequential(*modules)

    def input_prototype(self):
        return torch.randn(1, self.input_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass for generic feed-forward DNNs. Assumes activation names
        are valid pytorch activation names.
        :param input tensor
        """
        return self.dnn(input)


class FloatFeatureFullyConnected(ModelBase):
    """
    A fully connected network that takes FloatFeatures input
    and supports distributional prediction.
    """

    def __init__(
        self,
        state_dim,
        output_dim,
        sizes,
        activations,
        *,
        output_activation: str = "linear",
        num_atoms: Optional[int] = None,
        use_batch_norm: bool = False,
        dropout_ratio: float = 0.0,
        normalized_output: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        assert state_dim > 0, "state_dim must be > 0, got {}".format(state_dim)
        assert output_dim > 0, "output_dim must be > 0, got {}".format(output_dim)
        self.state_dim = state_dim
        self.output_dim = output_dim
        assert len(sizes) == len(
            activations
        ), "The numbers of sizes and activations must match; got {} vs {}".format(
            len(sizes), len(activations)
        )
        self.num_atoms = num_atoms
        self.fc = FullyConnectedNetwork(
            [state_dim] + sizes + [output_dim * (num_atoms or 1)],
            activations + [output_activation],
            use_batch_norm=use_batch_norm,
            dropout_ratio=dropout_ratio,
            normalize_output=normalized_output,
            use_layer_norm=use_layer_norm,
        )

    def input_prototype(self):
        return rlt.FeatureData(self.fc.input_prototype())

    def forward(
        self,
        state: rlt.FeatureData,
    ) -> torch.Tensor:
        float_features = state.float_features
        x = self.fc(float_features)
        if self.num_atoms is not None:
            x = x.view(float_features.shape[0], self.action_dim, self.num_atoms)
        return x
