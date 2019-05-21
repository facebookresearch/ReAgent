#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


logger = logging.getLogger(__name__)


def gaussian_fill_w_gain(tensor, activation, dim_in, min_std=0.0) -> None:
    """ Gaussian initialization with gain."""
    gain = math.sqrt(2) if activation == "relu" else 1
    init.normal_(tensor, mean=0, std=max(gain * math.sqrt(1 / dim_in), min_std))


class FullyConnectedNetwork(nn.Module):
    def __init__(
        self, layers, activations, use_batch_norm=False, min_std=0.0, dropout_ratio=0.0
    ) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        self.batch_norm_ops: nn.ModuleList = nn.ModuleList()
        self.activations = activations
        self.use_batch_norm = use_batch_norm
        self.dropout_layers: nn.ModuleList = nn.ModuleList()
        self.use_dropout = dropout_ratio > 0.0

        assert len(layers) >= 2, "Invalid layer schema {} for network".format(layers)

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            if self.use_batch_norm:
                self.batch_norm_ops.append(nn.BatchNorm1d(layers[i]))
            if self.use_dropout and i < len(layers[1:]) - 1:
                # applying dropout to all layers except
                # the input and the last output layer
                self.dropout_layers.append(nn.Dropout(p=dropout_ratio))
            gaussian_fill_w_gain(
                self.layers[i].weight, self.activations[i], layers[i], min_std
            )
            init.constant_(self.layers[i].bias, 0)

    def forward(self, input) -> torch.FloatTensor:
        """ Forward pass for generic feed-forward DNNs. Assumes activation names
        are valid pytorch activation names.
        :param input tensor
        """
        x = input
        for i, activation in enumerate(self.activations):
            if self.use_batch_norm:
                x = self.batch_norm_ops[i](x)
            x = self.layers[i](x)
            if activation == "linear":
                pass
            elif activation == "tanh":
                x = torch.tanh(x)
            else:
                x = getattr(F, activation)(x)
            if self.use_dropout and i < len(self.dropout_layers):
                x = self.dropout_layers[i](x)
        return x
