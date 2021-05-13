#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import collections
import logging
from typing import List, Optional

import torch
import torch.nn as nn
from reagent.core import parameters as rlp
from reagent.core import types as rlt
from reagent.models import convolutional_network
from reagent.models import fully_connected_network
from reagent.models.base import ModelBase
from reagent.models.fully_connected_network import ACTIVATION_MAP


logger = logging.getLogger(__name__)


class Concat(nn.Module):
    def forward(self, state: rlt.FeatureData, action: rlt.FeatureData):
        return torch.cat((state.float_features, action.float_features), dim=-1)


# pyre-fixme[11]: Annotation `Sequential` is not defined as a type.
class SequentialMultiArguments(nn.Sequential):
    """Sequential which can take more than 1 argument in forward function"""

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def _gen_mask(valid_step: torch.Tensor, batch_size: int, seq_len: int):
    """
    Mask for dealing with different lengths of MDPs

    Example:
    valid_step = [[1], [2], [3]], batch_size=3, seq_len = 4
    mask = [
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
    ]
    """
    assert valid_step.shape == (batch_size, 1)
    assert ((1 <= valid_step) <= seq_len).all()
    device = valid_step.device
    mask = torch.arange(seq_len, device=device).repeat(batch_size, 1)
    mask = (mask >= (seq_len - valid_step)).float()
    return mask


class SingleStepSyntheticRewardNet(ModelBase):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sizes: List[int],
        activations: List[str],
        last_layer_activation: str,
    ):
        """
        Decompose rewards at the last step to individual steps.
        """
        super().__init__()
        modules: List[nn.Module] = [Concat()]
        prev_layer_size = state_dim + action_dim
        for size, activation in zip(sizes, activations):
            modules.append(nn.Linear(prev_layer_size, size))
            modules.append(ACTIVATION_MAP[activation]())
            prev_layer_size = size
        # last layer
        modules.append(nn.Linear(prev_layer_size, 1))
        modules.append(ACTIVATION_MAP[last_layer_activation]())
        self.dnn = SequentialMultiArguments(*modules)

    def forward(self, training_batch: rlt.MemoryNetworkInput):
        # state shape: seq_len, batch_size, state_dim
        state = training_batch.state
        # action shape: seq_len, batch_size, action_dim
        action = rlt.FeatureData(float_features=training_batch.action)

        # shape: batch_size, 1
        valid_step = training_batch.valid_step
        seq_len, batch_size, _ = training_batch.action.shape

        # output shape: batch_size, seq_len
        # pyre-fixme[29]: `SequentialMultiArguments` is not a function.
        output = self.dnn(state, action).squeeze(2).transpose(0, 1)
        assert valid_step is not None
        mask = _gen_mask(valid_step, batch_size, seq_len)
        output *= mask

        pred_reward = output.sum(dim=1, keepdim=True)
        return rlt.RewardNetworkOutput(predicted_reward=pred_reward)

    def export_mlp(self):
        return self.dnn


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


class NGramConvolutionalNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sizes: List[int],
        activations: List[str],
        last_layer_activation: str,
        context_size: int,
        conv_net_params: rlp.ConvNetParameters,
    ) -> None:
        super().__init__()

        self.input_width = state_dim + action_dim
        self.input_height = context_size
        self.num_input_channels = 1

        num_conv_layers = len(conv_net_params.conv_height_kernels)
        conv_width_kernels = [self.input_width] + [1] * (num_conv_layers - 1)

        cnn_parameters = CnnParameters(
            conv_dims=[self.num_input_channels] + conv_net_params.conv_dims,
            conv_height_kernels=conv_net_params.conv_height_kernels,
            conv_width_kernels=conv_width_kernels,
            pool_types=conv_net_params.pool_types,
            pool_kernels_strides=conv_net_params.pool_kernel_sizes,
            num_input_channels=self.num_input_channels,
            input_height=self.input_height,
            input_width=self.input_width,
        )

        self.conv_net = convolutional_network.ConvolutionalNetwork(
            cnn_parameters, [-1] + sizes + [1], activations + [last_layer_activation]
        )

    def forward(self, input) -> torch.FloatTensor:
        """Forward pass NGram conv net.

        :param input shape: seq_len, batch_size, feature_dim
        """
        # shape: seq_len * batch_size, 1, context_size, state_dim + action_dim
        seq_len, batch_size, _ = input.shape
        reshaped = input.reshape(-1, 1, self.input_height, self.input_width)
        # shape: seq_len * batch_size, 1
        output = self.conv_net.forward(reshaped)
        # shape: seq_len, batch_size, 1
        return output.reshape(seq_len, batch_size, 1)


class NGramSyntheticRewardNet(ModelBase):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sizes: List[int],
        activations: List[str],
        last_layer_activation: str,
        context_size: int,
        use_batch_norm: bool = False,  # not supported for conv net yet
        use_layer_norm: bool = False,  # not supported for conv net yet
        conv_net_params: Optional[rlp.ConvNetParameters] = None,
    ):
        """
        Decompose rewards at the last step to individual steps.
        """
        super().__init__()

        assert context_size % 2 == 1, f"Context size is not odd: {context_size}"

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_size = context_size

        self.ngram_padding = torch.zeros(1, 1, state_dim + action_dim)

        if conv_net_params is None:
            self.net = fully_connected_network.FullyConnectedNetwork(
                [(state_dim + action_dim) * context_size] + sizes + [1],
                activations + [last_layer_activation],
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
            )
        else:
            self.net = NGramConvolutionalNetwork(
                state_dim,
                action_dim,
                sizes,
                activations,
                last_layer_activation,
                context_size,
                conv_net_params,
            )

    def _ngram(self, input):
        seq_len, batch_size, feature_dim = input.shape

        shifted_list = []
        for i in range(self.context_size):
            offset = i - self.context_size // 2
            if offset < 0:
                shifted = torch.cat(
                    (
                        self.ngram_padding.tile((-offset, batch_size, 1)),
                        input.narrow(0, 0, seq_len + offset),
                    ),
                    dim=0,
                )
            elif offset > 0:
                shifted = torch.cat(
                    (
                        input.narrow(0, offset, seq_len - offset),
                        self.ngram_padding.tile(offset, batch_size, 1),
                    ),
                    dim=0,
                )
            else:
                shifted = input
            shifted_list.append(shifted)

        # shape: seq_len, batch_size, feature_dim * context_size
        return torch.cat(shifted_list, -1)

    def forward(self, training_batch: rlt.MemoryNetworkInput):
        # state shape: seq_len, batch_size, state_dim
        state = training_batch.state
        # action shape: seq_len, batch_size, action_dim
        action = rlt.FeatureData(float_features=training_batch.action)

        # shape: seq_len, batch_size, state_dim + action_dim
        cat_input = torch.cat((state.float_features, action.float_features), dim=-1)

        # shape: seq_len, batch_size, (state_dim + action_dim) * context_size
        ngram = self._ngram(cat_input)

        # shape: batch_size, 1
        valid_step = training_batch.valid_step
        seq_len, batch_size, _ = training_batch.action.shape

        # output shape: batch_size, seq_len
        output = self.net(ngram).squeeze(2).transpose(0, 1)
        assert valid_step is not None
        mask = _gen_mask(valid_step, batch_size, seq_len)
        output *= mask

        pred_reward = output.sum(dim=1, keepdim=True)
        return rlt.RewardNetworkOutput(predicted_reward=pred_reward)
