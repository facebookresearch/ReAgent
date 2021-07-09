#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from reagent.core import parameters as rlp
from reagent.core import types as rlt
from reagent.models import convolutional_network
from reagent.models import fully_connected_network
from reagent.models.base import ModelBase
from reagent.models.fully_connected_network import ACTIVATION_MAP

logger = logging.getLogger(__name__)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


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


class ResidualBlock(nn.Module):
    def __init__(self, d_model=64, dim_feedforward=128):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU()
        self.fc_residual = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.fc_residual(x))


class PositionalEncoding(nn.Module):
    def __init__(self, feature_dim=128, dropout=0.0, max_len=100):
        """
        This module injects some information about the relative or absolute position of the tokens in the sequence.
        The generated positional encoding are concatenated together with the features.
        Args: input dim
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, feature_dim, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2).float() * (-math.log(10000.0) / feature_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # max_len * feature_dim // 2
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe dimension: (max_len, 1, feature_dim)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x dimension: (L, B, E)
        # batch_size, seq_len, d_model
        seq_len = x.shape[0]
        pos_encoding = self.pe[:seq_len, :]
        x = x + pos_encoding
        return self.dropout(x)


class PETransformerEncoderLayer(nn.Module):
    """PETransformerEncoderLayer is made up of Positional Encoding (PE), residual connections, self-attn and feedforward network.
    Major differences between this implementation and the pytorch official torch.nn.TransformerEncoderLayer are:
    1. Augment input data with positional encoding. hat{x} = x + PE{x}
    2. Two paralle residual blocks are applied to the raw input data (x) and encoded input data (hat{x}), respectively, i.e. z = Residual(x), hat{z} = Residual(hat{x})
    3. Treat z as the Value input, and hat{z} as the Query and Key input to feed a self-attention block.

    Main Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.
        max_len: argument passed to the Positional Encoding module, see more details in the PositionalEncoding class.
    """

    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        layer_norm_eps=1e-5,
        max_len=100,
        use_ff=True,
        pos_weight=0.5,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(PETransformerEncoderLayer, self).__init__()
        self.use_ff = use_ff
        self.pos_weight = pos_weight
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Customized implementation: to map Query & Key, Value with different embeddings.
        self.qk_residual = ResidualBlock(d_model, dim_feedforward)
        self.v_residual = ResidualBlock(d_model, dim_feedforward)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(PETransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        encoded_src = self.pos_encoder(src)
        query = self.qk_residual(encoded_src)
        # do not involve pos_encoding info into the value
        src = self.v_residual(src)

        src2 = self.self_attn(
            query,  # query
            query,  # key = query as the input
            src,  # value
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        # add transformer related residual
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # add another ff layer
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


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
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        """
        Decompose rewards at the last step to individual steps.
        """
        super().__init__()
        modules: List[nn.Module] = [Concat()]
        prev_layer_size = state_dim + action_dim
        for size, activation in zip(sizes, activations):
            if use_batch_norm:
                modules.append(nn.BatchNorm1d(prev_layer_size))
            modules.append(nn.Linear(prev_layer_size, size))
            if use_layer_norm:
                modules.append(nn.LayerNorm(size))
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
        use_layer_norm: bool = False,
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
            cnn_parameters,
            [-1] + sizes + [1],
            activations + [last_layer_activation],
            use_layer_norm=use_layer_norm,
        )

        self.ngram_padding = torch.zeros(1, 1, state_dim + action_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass NGram conv net.

        :param input shape: seq_len, batch_size, feature_dim
        """
        # shape: seq_len, batch_size, state_dim + action_dim
        input = torch.cat((state, action), dim=-1)
        # shape: seq_len, batch_size, (state_dim + action_dim) * context_size
        ngram_input = ngram(
            input, self.context_size, self.ngram_padding.to(input.device)
        )

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
        use_layer_norm: bool = False,
    ) -> None:
        assert context_size % 2 == 1, f"Context size is not odd: {context_size}"
        super().__init__()
        self.context_size = context_size
        self.ngram_padding = torch.zeros(1, 1, state_dim + action_dim)
        self.fc = fully_connected_network.FullyConnectedNetwork(
            [(state_dim + action_dim) * context_size] + sizes + [1],
            activations + [last_layer_activation],
            use_layer_norm=use_layer_norm,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass NGram conv net.

        :param input shape: seq_len, batch_size, feature_dim
        """
        input = torch.cat((state, action), dim=-1)
        # shape: seq_len, batch_size, (state_dim + action_dim) * context_size
        ngram_input = ngram(
            input, self.context_size, self.ngram_padding.to(input.device)
        )
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


class TransformerSyntheticRewardNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int,
        nhead: int = 2,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        activation: str = "relu",
        last_layer_activation: str = "leaky_relu",
        layer_norm_eps: float = 1e-5,
        max_len: int = 10,
    ):
        """
        Decompose rewards at the last step to individual steps using transformer modules.

        Args:
            nhead: the number of heads in the multiheadattention models (default=8).
            num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
            activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
            layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        # d_model: dimension of transformer input
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.max_len = max_len

        # map input features to higher latent space before sending to transformer
        self.fc_in = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.d_model),
            nn.ReLU(),
        )

        # use transformer encoder to get reward logits for each step
        encoder_layer = PETransformerEncoderLayer(
            self.d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            max_len=self.max_len,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
        )
        self.fc_out = nn.Linear(self.d_model, 1)
        self.output_activation = ACTIVATION_MAP[last_layer_activation]()

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        # shape: seq_len (L), batch_size (B), state_dim + action_dim
        cat_input = torch.cat((state, action), dim=-1)
        # latent_input shape: (L,B,E)
        latent_input = self.fc_in(cat_input)
        # output shape: (L, B, E)
        output = self.transformer(latent_input)
        output = self.fc_out(output)
        # output shape: seq_len, batch_size, 1
        output = self.output_activation(output).squeeze(2).transpose(0, 1)
        # output shape: batch_size, seq_len
        return output
