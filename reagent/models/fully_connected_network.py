#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.init as init
from reagent.core import types as rlt
from reagent.models.base import ModelBase


logger = logging.getLogger(__name__)


def gaussian_fill_w_gain(tensor, gain, dim_in, min_std=0.0) -> None:
    """Gaussian initialization with gain."""
    init.normal_(tensor, mean=0, std=max(gain * math.sqrt(1 / dim_in), min_std))


ACTIVATION_MAP = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "linear": nn.Identity,
    "sigmoid": nn.Sigmoid,
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
        assert len(x.shape) in [2, 3], f"Invalid input shape {x.shape}"
        if len(x.shape) == 2:
            return self.vanilla(x)
        if len(x.shape) == 3:
            return self.vanilla(x.transpose(1, 2)).transpose(1, 2)


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
                modules.append(SlateBatchNorm1d(in_dim))
            # Add Linear
            linear = nn.Linear(in_dim, out_dim)
            # assuming activation is valid
            gain = torch.nn.init.calculate_gain(activation)
            if orthogonal_init:
                # provably better https://openreview.net/forum?id=rkgqN1SYvr
                nn.init.orthogonal_(linear.weight.data, gain=gain)
            else:
                # gaussian init
                gaussian_fill_w_gain(
                    linear.weight, gain=gain, dim_in=in_dim, min_std=min_std
                )

            init.constant_(linear.bias, 0)  # type: ignore
            modules.append(linear)
            # Add LayerNorm
            if use_layer_norm and (normalize_output or i < len(activations) - 1):
                modules.append(nn.LayerNorm(out_dim))  # type: ignore
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
            activations + ["linear"],
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
