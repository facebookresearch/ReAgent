#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

import torch
from reagent.core import parameters as rlp
from reagent.models import synthetic_reward
from reagent.models.synthetic_reward import NGramSyntheticRewardNet
from reagent.models.synthetic_reward import SingleStepSyntheticRewardNet


logger = logging.getLogger(__name__)


class TestSyntheticReward(unittest.TestCase):
    def test_single_step_synthetic_reward(self):
        state_dim = 10
        action_dim = 2
        sizes = [256, 128]
        activations = ["sigmoid", "relu"]
        last_layer_activation = "leaky_relu"
        reward_net = SingleStepSyntheticRewardNet(
            state_dim=state_dim,
            action_dim=action_dim,
            sizes=sizes,
            activations=activations,
            last_layer_activation=last_layer_activation,
        )
        dnn = reward_net.export_mlp()
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
        mask = synthetic_reward._gen_mask(valid_step, batch_size, seq_len)
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

        fc = synthetic_reward.NGramFullyConnectedNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            sizes=sizes,
            activations=activations,
            last_layer_activation=last_layer_activation,
            context_size=context_size,
        )

        reward_net = NGramSyntheticRewardNet(
            state_dim=state_dim,
            action_dim=action_dim,
            context_size=context_size,
            net=fc,
        )

        dnn = reward_net.net.fc.dnn
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
        mask = synthetic_reward._gen_mask(valid_step, batch_size, seq_len)
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
        conv_net = synthetic_reward.NGramConvolutionalNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            sizes=sizes,
            activations=activations,
            last_layer_activation=last_layer_activation,
            context_size=context_size,
            conv_net_params=conv_net_params,
        )

        reward_net = NGramSyntheticRewardNet(
            state_dim=state_dim,
            action_dim=action_dim,
            context_size=context_size,
            net=conv_net,
        )
        conv_net = reward_net.net.conv_net

        assert conv_net.conv_dims == [1, 256, 128]
        assert conv_net.conv_height_kernels == [1, 1]
        assert conv_net.conv_width_kernels == [12, 1]

        dnn = conv_net.feed_forward.dnn
        assert dnn[0].in_features == 384
        assert dnn[0].out_features == 256
        assert dnn[1]._get_name() == "Sigmoid"
        assert dnn[2].in_features == 256
        assert dnn[2].out_features == 128
        assert dnn[3]._get_name() == "ReLU"
        assert dnn[4].in_features == 128
        assert dnn[4].out_features == 1
        assert dnn[5]._get_name() == "LeakyReLU"
