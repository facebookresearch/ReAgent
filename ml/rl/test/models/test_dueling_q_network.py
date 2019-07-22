#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

from ml.rl.models.dueling_q_network import DuelingQNetwork
from ml.rl.test.models.test_utils import check_save_load


logger = logging.getLogger(__name__)


class TestDuelingQNetwork(unittest.TestCase):
    def test_discrete_action(self):
        state_dim = 8
        action_dim = 4
        model = DuelingQNetwork(
            layers=[state_dim, 8, 4, action_dim],
            activations=["relu", "relu", "linear"],
            use_batch_norm=True,
        )
        input = model.input_prototype()
        self.assertEqual((1, state_dim), input.state.float_features.shape)
        # Using batch norm requires more than 1 example in training, avoid that
        model.eval()
        q_values = model(input)
        self.assertEqual((1, action_dim), q_values.q_values.shape)

    def test_parametric_action(self):
        state_dim = 8
        action_dim = 4
        model = DuelingQNetwork(
            layers=[state_dim, 8, 4, 1],
            activations=["relu", "relu", "linear"],
            use_batch_norm=True,
            action_dim=action_dim,
        )
        input = model.input_prototype()
        self.assertEqual((1, state_dim), input.state.float_features.shape)
        self.assertEqual((1, action_dim), input.action.float_features.shape)
        # Using batch norm requires more than 1 example in training, avoid that
        model.eval()
        q_values = model(input)
        self.assertEqual((1, 1), q_values.q_value.shape)

    def test_save_load_discrete_action(self):
        state_dim = 8
        action_dim = 4
        model = DuelingQNetwork(
            layers=[state_dim, 8, 4, action_dim],
            activations=["relu", "relu", "linear"],
            use_batch_norm=False,
        )
        expected_num_params, expected_num_inputs, expected_num_outputs = 22, 1, 1
        check_save_load(
            self, model, expected_num_params, expected_num_inputs, expected_num_outputs
        )

    def test_save_load_parametric_action(self):
        state_dim = 8
        action_dim = 4
        model = DuelingQNetwork(
            layers=[state_dim, 8, 4, 1],
            activations=["relu", "relu", "linear"],
            use_batch_norm=False,
            action_dim=action_dim,
        )
        expected_num_params, expected_num_inputs, expected_num_outputs = 22, 2, 1
        check_save_load(
            self, model, expected_num_params, expected_num_inputs, expected_num_outputs
        )

    def test_save_load_discrete_action_batch_norm(self):
        state_dim = 8
        action_dim = 4
        model = DuelingQNetwork(
            layers=[state_dim, 8, 4, action_dim],
            activations=["relu", "relu", "linear"],
            use_batch_norm=False,
        )
        # Freezing batch_norm
        model.eval()
        # Number of expected params is the same because DuelingQNetwork always
        # initialize batch norm layer even if it doesn't use it.
        expected_num_params, expected_num_inputs, expected_num_outputs = 22, 1, 1
        check_save_load(
            self, model, expected_num_params, expected_num_inputs, expected_num_outputs
        )

    def test_save_load_parametric_action_batch_norm(self):
        state_dim = 8
        action_dim = 4
        model = DuelingQNetwork(
            layers=[state_dim, 8, 4, 1],
            activations=["relu", "relu", "linear"],
            use_batch_norm=False,
            action_dim=action_dim,
        )
        # Freezing batch_norm
        model.eval()
        # Number of expected params is the same because DuelingQNetwork always
        # initialize batch norm layer even if it doesn't use it.
        expected_num_params, expected_num_inputs, expected_num_outputs = 22, 2, 1
        check_save_load(
            self, model, expected_num_params, expected_num_inputs, expected_num_outputs
        )
