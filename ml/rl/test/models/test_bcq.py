#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

import torch
from ml.rl.models.bcq import BatchConstrainedDQN
from ml.rl.models.dqn import FullyConnectedDQN
from ml.rl.models.fully_connected_network import FullyConnectedNetwork
from ml.rl.test.models.test_utils import check_save_load
from ml.rl.types import FeatureVector, StateInput


logger = logging.getLogger(__name__)


class TestBCQ(unittest.TestCase):
    def test_basic(self):
        state_dim = 8
        action_dim = 4
        q_network = FullyConnectedDQN(
            state_dim, action_dim, sizes=[8, 4], activations=["relu", "relu"]
        )
        imitator_network = FullyConnectedNetwork(
            layers=[state_dim, 8, 4, action_dim], activations=["relu", "relu", "linear"]
        )
        model = BatchConstrainedDQN(
            state_dim=state_dim,
            q_network=q_network,
            imitator_network=imitator_network,
            bcq_drop_threshold=0.05,
        )

        input = model.input_prototype()
        self.assertEqual((1, state_dim), input.state.float_features.shape)
        q_values = model(input)
        self.assertEqual((1, action_dim), q_values.q_values.shape)

    def test_save_load(self):
        state_dim = 8
        action_dim = 4
        q_network = FullyConnectedDQN(
            state_dim, action_dim, sizes=[8, 4], activations=["relu", "relu"]
        )
        imitator_network = FullyConnectedNetwork(
            layers=[state_dim, 8, 4, action_dim], activations=["relu", "relu", "linear"]
        )
        model = BatchConstrainedDQN(
            state_dim=state_dim,
            q_network=q_network,
            imitator_network=imitator_network,
            bcq_drop_threshold=0.05,
        )
        # 6 for DQN + 6 for Imitator Network + 2 for BCQ constants
        expected_num_params, expected_num_inputs, expected_num_outputs = 14, 1, 1
        check_save_load(
            self, model, expected_num_params, expected_num_inputs, expected_num_outputs
        )

    def test_forward_pass(self):
        state_dim = 1
        action_dim = 2
        input = StateInput(state=FeatureVector(float_features=torch.tensor([[2.0]])))
        bcq_drop_threshold = 0.20

        q_network = FullyConnectedDQN(
            state_dim, action_dim, sizes=[2], activations=["relu"]
        )
        # Set weights of q-network to make it deterministic
        q_net_layer_0_w = torch.tensor([[1.2], [0.9]])
        q_network.state_dict()["fc.layers.0.weight"].data.copy_(q_net_layer_0_w)
        q_net_layer_0_b = torch.tensor([0.0, 0.0])
        q_network.state_dict()["fc.layers.0.bias"].data.copy_(q_net_layer_0_b)
        q_net_layer_1_w = torch.tensor([[0.5, -0.5], [1.0, 1.0]])
        q_network.state_dict()["fc.layers.1.weight"].data.copy_(q_net_layer_1_w)
        q_net_layer_1_b = torch.tensor([0.0, 0.0])
        q_network.state_dict()["fc.layers.1.bias"].data.copy_(q_net_layer_1_b)

        imitator_network = FullyConnectedNetwork(
            layers=[state_dim, 2, action_dim], activations=["relu", "linear"]
        )
        # Set weights of imitator network to make it deterministic
        im_net_layer_0_w = torch.tensor([[1.2], [0.9]])
        imitator_network.state_dict()["layers.0.weight"].data.copy_(im_net_layer_0_w)
        im_net_layer_0_b = torch.tensor([0.0, 0.0])
        imitator_network.state_dict()["layers.0.bias"].data.copy_(im_net_layer_0_b)
        im_net_layer_1_w = torch.tensor([[0.5, 1.5], [1.0, 2.0]])
        imitator_network.state_dict()["layers.1.weight"].data.copy_(im_net_layer_1_w)
        im_net_layer_1_b = torch.tensor([0.0, 0.0])
        imitator_network.state_dict()["layers.1.bias"].data.copy_(im_net_layer_1_b)

        imitator_probs = torch.nn.functional.softmax(
            imitator_network(input.state.float_features), dim=1
        )
        bcq_mask = imitator_probs < bcq_drop_threshold
        assert bcq_mask[0][0] == 1
        assert bcq_mask[0][1] == 0

        model = BatchConstrainedDQN(
            state_dim=state_dim,
            q_network=q_network,
            imitator_network=imitator_network,
            bcq_drop_threshold=bcq_drop_threshold,
        )
        final_q_values = model(input)
        assert final_q_values.q_values[0][0] == -1e10
        assert abs(final_q_values.q_values[0][1] - 4.2) < 0.0001
