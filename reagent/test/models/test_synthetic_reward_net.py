#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

import torch
from reagent.core import parameters as rlp
from reagent.models.synthetic_reward import (
    SingleStepSyntheticRewardNet,
    SequenceSyntheticRewardNet,
    TransformerSyntheticRewardNet,
    NGramFullyConnectedNetwork,
    NGramConvolutionalNetwork,
    SyntheticRewardNet,
    _gen_mask,
)


logger = logging.getLogger(__name__)


class TestSyntheticReward(unittest.TestCase):
    def test_single_step_synthetic_reward(self):
        state_dim = 10
        action_dim = 2
        sizes = [256, 128]
        activations = ["sigmoid", "relu"]
        last_layer_activation = "leaky_relu"
        reward_net = SyntheticRewardNet(
            SingleStepSyntheticRewardNet(
                state_dim=state_dim,
                action_dim=action_dim,
                sizes=sizes,
                activations=activations,
                last_layer_activation=last_layer_activation,
            )
        )
        dnn = reward_net.export_mlp().dnn
        # dnn[0] is a concat layer
        assert dnn[1].in_features == state_dim + action_dim
        assert dnn[1].out_features == 256
        assert dnn[2]._get_name() == "Sigmoid"
        assert dnn[3].in_features == 256
        assert dnn[3].out_features == 128
        assert dnn[4]._get_name() == "ReLU"
        assert dnn[5].in_features == 128
        assert dnn[5].out_features == 1
        assert dnn[6]._get_name() == "LeakyReLU"

        valid_step = torch.tensor([[1], [2], [3]])
        batch_size = 3
        seq_len = 4
        mask = _gen_mask(valid_step, batch_size, seq_len)
        assert torch.all(
            mask
            == torch.tensor(
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]]
            )
        )

    def test_ngram_fc_synthetic_reward(self):
        state_dim = 10
        action_dim = 2
        sizes = [256, 128]
        activations = ["sigmoid", "relu"]
        last_layer_activation = "leaky_relu"
        context_size = 3

        net = NGramFullyConnectedNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            sizes=sizes,
            activations=activations,
            last_layer_activation=last_layer_activation,
            context_size=context_size,
        )
        reward_net = SyntheticRewardNet(net)

        dnn = reward_net.export_mlp().fc.dnn
        assert dnn[0].in_features == (state_dim + action_dim) * context_size
        assert dnn[0].out_features == 256
        assert dnn[1]._get_name() == "Sigmoid"
        assert dnn[2].in_features == 256
        assert dnn[2].out_features == 128
        assert dnn[3]._get_name() == "ReLU"
        assert dnn[4].in_features == 128
        assert dnn[4].out_features == 1
        assert dnn[5]._get_name() == "LeakyReLU"

        valid_step = torch.tensor([[1], [2], [3]])
        batch_size = 3
        seq_len = 4
        mask = _gen_mask(valid_step, batch_size, seq_len)
        assert torch.all(
            mask
            == torch.tensor(
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]]
            )
        )

    def test_ngram_conv_net_synthetic_reward(self):
        state_dim = 10
        action_dim = 2
        sizes = [256, 128]
        activations = ["sigmoid", "relu"]
        last_layer_activation = "leaky_relu"
        context_size = 3

        conv_net_params = rlp.ConvNetParameters(
            conv_dims=[256, 128],
            conv_height_kernels=[1, 1],
            pool_types=["max", "max"],
            pool_kernel_sizes=[1, 1],
        )
        net = NGramConvolutionalNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            sizes=sizes,
            activations=activations,
            last_layer_activation=last_layer_activation,
            context_size=context_size,
            conv_net_params=conv_net_params,
        )

        reward_net = SyntheticRewardNet(net)
        conv_net = reward_net.export_mlp().conv_net

        assert conv_net.conv_dims == [1, 256, 128]
        assert conv_net.conv_height_kernels == [1, 1]
        assert conv_net.conv_width_kernels == [12, 1]

        assert conv_net.conv_layers[0].in_channels == 1
        assert conv_net.conv_layers[0].out_channels == 256
        assert conv_net.conv_layers[0].kernel_size == (1, 12)
        assert conv_net.conv_layers[0].stride == (1, 1)
        assert conv_net.conv_layers[1].in_channels == 256
        assert conv_net.conv_layers[1].out_channels == 128
        assert conv_net.conv_layers[1].kernel_size == (1, 1)
        assert conv_net.conv_layers[1].stride == (1, 1)

        dnn = reward_net.export_mlp().conv_net.feed_forward.dnn
        assert dnn[0].in_features == 384
        assert dnn[0].out_features == 256
        assert dnn[1]._get_name() == "Sigmoid"
        assert dnn[2].in_features == 256
        assert dnn[2].out_features == 128
        assert dnn[3]._get_name() == "ReLU"
        assert dnn[4].in_features == 128
        assert dnn[4].out_features == 1
        assert dnn[5]._get_name() == "LeakyReLU"

    def test_lstm_synthetic_reward(self):
        state_dim = 10
        action_dim = 2
        last_layer_activation = "leaky_relu"
        net = SequenceSyntheticRewardNet(
            state_dim=state_dim,
            action_dim=action_dim,
            lstm_hidden_size=128,
            lstm_num_layers=2,
            lstm_bidirectional=True,
            last_layer_activation=last_layer_activation,
        )
        reward_net = SyntheticRewardNet(net)
        lstm = reward_net.export_mlp().lstm
        assert lstm.bidirectional
        assert lstm.input_size == 12
        assert lstm.hidden_size == 128
        assert lstm.num_layers == 2

        dnn = reward_net.export_mlp().fc_out
        assert dnn.in_features == 128 * 2
        assert dnn.out_features == 1

        output_activation = reward_net.export_mlp().output_activation
        assert output_activation._get_name() == "LeakyReLU"

    def test_transformer_synthetic_reward(self):
        state_dim = 10
        action_dim = 2
        d_model = 64
        nhead = 8
        num_encoder_layers = 2
        dim_feedforward = 64
        dropout = 0.0
        activation = "relu"
        last_layer_activation = "leaky_relu"
        layer_norm_eps = 1e-5
        max_len = 10

        net = TransformerSyntheticRewardNet(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            last_layer_activation=last_layer_activation,
            layer_norm_eps=layer_norm_eps,
            max_len=max_len,
        )

        reward_net = SyntheticRewardNet(net)
        export_net = reward_net.export_mlp()
        transformer = export_net.transformer
        assert export_net.state_dim == state_dim
        assert export_net.action_dim == action_dim
        assert export_net.d_model == d_model
        assert export_net.nhead == nhead
        assert export_net.dim_feedforward == dim_feedforward
        assert export_net.dropout == dropout
        assert export_net.activation == activation
        assert export_net.layer_norm_eps == layer_norm_eps

        assert transformer.num_layers == num_encoder_layers
        dnn_out = export_net.fc_out
        assert dnn_out.in_features == d_model
        assert dnn_out.out_features == 1

        output_activation = export_net.output_activation
        assert output_activation._get_name() == "LeakyReLU"
