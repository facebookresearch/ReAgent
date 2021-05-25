#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import List

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
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        return torch.cat((state, action), dim=-1)


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


def ngram(input: torch.Tensor, context_size: int, ngram_padding: torch.Tensor):
    # input shape: seq_len, batch_size, state_dim + action_dim
    seq_len, batch_size, feature_dim = input.shape

    shifted_list = []
    for i in range(context_size):
        offset = i - context_size // 2
        if offset < 0:
            shifted = torch.cat(
                (
                    # pyre-fixme[16]: `Tensor` has no attribute `tile`.
                    ngram_padding.tile((-offset, batch_size, 1)),
                    # pyre-fixme[16]: `Tensor` has no attribute `narrow`.
                    input.narrow(0, 0, seq_len + offset),
                ),
                dim=0,
            )
        elif offset > 0:
            shifted = torch.cat(
                (
                    input.narrow(0, offset, seq_len - offset),
                    ngram_padding.tile(offset, batch_size, 1),
                ),
                dim=0,
            )
        else:
            shifted = input
        shifted_list.append(shifted)

    # shape: seq_len, batch_size, feature_dim * context_size
    return torch.cat(shifted_list, dim=-1)


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
    assert (1 <= valid_step).all()
    assert (valid_step <= seq_len).all()
    device = valid_step.device
    mask = torch.arange(seq_len, device=device).repeat(batch_size, 1)
    mask = (mask >= (seq_len - valid_step)).float()
    return mask


class SyntheticRewardNet(ModelBase):
    """
    This base class provides basic operations to consume inputs and call a synthetic reward net

    A synthetic reward net (self.net) assumes the input contains only torch.Tensors.
    Expected input shape:
        state: seq_len, batch_size, state_dim
        action: seq_len, batch_size, action_dim
    Expected output shape:
        reward: batch_size, seq_len
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, training_batch: rlt.MemoryNetworkInput):
        # state shape: seq_len, batch_size, state_dim
        state = training_batch.state.float_features
        # action shape: seq_len, batch_size, action_dim
        action = training_batch.action

        # shape: batch_size, 1
        valid_step = training_batch.valid_step
        seq_len, batch_size, _ = training_batch.action.shape

        # output shape: batch_size, seq_len
        output = self.net(state, action)
        assert valid_step is not None
        mask = _gen_mask(valid_step, batch_size, seq_len)
        output_masked = output * mask

        pred_reward = output_masked.sum(dim=1, keepdim=True)
        return rlt.RewardNetworkOutput(predicted_reward=pred_reward)

    def export_mlp(self):
        """
        Export an pytorch nn to feed to predictor wrapper.
        """
        return self.net


class SingleStepSyntheticRewardNet(nn.Module):
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

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        # pyre-fixme[29]: `SequentialMultiArguments` is not a function.
        # shape: batch_size, seq_len
        return self.dnn(state, action).squeeze(2).transpose(0, 1)


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
        assert context_size % 2 == 1, f"Context size is not odd: {context_size}"
        super().__init__()

        self.context_size = context_size
        self.input_width = state_dim + action_dim
        self.input_height = context_size
        self.num_input_channels = 1

        num_conv_layers = len(conv_net_params.conv_height_kernels)
        conv_width_kernels = [self.input_width] + [1] * (num_conv_layers - 1)

        cnn_parameters = convolutional_network.CnnParameters(
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

        self.ngram_padding = torch.zeros(1, 1, state_dim + action_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass NGram conv net.

        :param input shape: seq_len, batch_size, feature_dim
        """
        # shape: seq_len, batch_size, state_dim + action_dim
        input = torch.cat((state, action), dim=-1)
        # shape: seq_len, batch_size, (state_dim + action_dim) * context_size
        ngram_input = ngram(input, self.context_size, self.ngram_padding)

        seq_len, batch_size, _ = ngram_input.shape
        # shape: seq_len * batch_size, 1, context_size, state_dim + action_dim
        reshaped = ngram_input.reshape(-1, 1, self.input_height, self.input_width)
        # shape: batch_size, seq_len
        output = self.conv_net(reshaped).reshape(seq_len, batch_size).transpose(0, 1)
        return output


class NGramFullyConnectedNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sizes: List[int],
        activations: List[str],
        last_layer_activation: str,
        context_size: int,
    ) -> None:
        assert context_size % 2 == 1, f"Context size is not odd: {context_size}"
        super().__init__()
        self.context_size = context_size
        self.ngram_padding = torch.zeros(1, 1, state_dim + action_dim)
        self.fc = fully_connected_network.FullyConnectedNetwork(
            [(state_dim + action_dim) * context_size] + sizes + [1],
            activations + [last_layer_activation],
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass NGram conv net.

        :param input shape: seq_len, batch_size, feature_dim
        """
        input = torch.cat((state, action), dim=-1)
        # shape: seq_len, batch_size, (state_dim + action_dim) * context_size
        ngram_input = ngram(input, self.context_size, self.ngram_padding)
        # shape: batch_size, seq_len
        return self.fc(ngram_input).transpose(0, 1).squeeze(2)


class SequenceSyntheticRewardNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        lstm_bidirectional: bool,
        last_layer_activation: str,
    ):
        """
        Decompose rewards at the last step to individual steps.
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bidirectional = lstm_bidirectional

        self.lstm = nn.LSTM(
            input_size=self.state_dim + self.action_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            bidirectional=self.lstm_bidirectional,
        )

        if self.lstm_bidirectional:
            self.fc_out = nn.Linear(self.lstm_hidden_size * 2, 1)
        else:
            self.fc_out = nn.Linear(self.lstm_hidden_size, 1)

        self.output_activation = ACTIVATION_MAP[last_layer_activation]()

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        # shape: seq_len, batch_size, state_dim + action_dim
        cat_input = torch.cat((state, action), dim=-1)
        # output shape: seq_len, batch_size, self.hidden_size
        output, _ = self.lstm(cat_input)
        # output shape: seq_len, batch_size, 1
        output = self.fc_out(output)
        # output shape: batch_size, seq_len
        output = self.output_activation(output).squeeze(2).transpose(0, 1)
        return output
