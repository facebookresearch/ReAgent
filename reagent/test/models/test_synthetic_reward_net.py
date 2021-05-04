#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

import torch
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
        mask = reward_net.gen_mask(valid_step, batch_size, seq_len)
        assert torch.all(
            mask
            == torch.tensor(
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]]
            )
        )
