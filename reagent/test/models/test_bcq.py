#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

import numpy.testing as npt
import torch
import torch.nn.init as init
from reagent import types as rlt
from reagent.models.bcq import BatchConstrainedDQN
from reagent.models.dqn import FullyConnectedDQN
from reagent.models.fully_connected_network import FullyConnectedNetwork
from reagent.test.models.test_utils import check_save_load


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
        self.assertEqual((1, state_dim), input.float_features.shape)
        q_values = model(input)
        self.assertEqual((1, action_dim), q_values.shape)

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
        torch.manual_seed(123)
        state_dim = 1
        action_dim = 2
        state = rlt.FeatureData(torch.tensor([[2.0]]))
        bcq_drop_threshold = 0.20

        q_network = FullyConnectedDQN(
            state_dim, action_dim, sizes=[2], activations=["relu"]
        )
        init.constant_(q_network.fc.dnn[-2].bias, 3.0)
        imitator_network = FullyConnectedNetwork(
            layers=[state_dim, 2, action_dim], activations=["relu", "linear"]
        )

        imitator_probs = torch.nn.functional.softmax(
            imitator_network(state.float_features), dim=1
        )
        bcq_mask = imitator_probs < bcq_drop_threshold
        npt.assert_array_equal(bcq_mask.detach(), [[True, False]])

        model = BatchConstrainedDQN(
            state_dim=state_dim,
            q_network=q_network,
            imitator_network=imitator_network,
            bcq_drop_threshold=bcq_drop_threshold,
        )
        final_q_values = model(state)
        npt.assert_array_equal(final_q_values.detach(), [[-1e10, 3.0]])
