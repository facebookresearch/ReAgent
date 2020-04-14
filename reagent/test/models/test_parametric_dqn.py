#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

from ml.rl.models.parametric_dqn import FullyConnectedParametricDQN
from ml.rl.test.models.test_utils import check_save_load


logger = logging.getLogger(__name__)


class TestFullyConnectedParametricDQN(unittest.TestCase):
    def test_basic(self):
        state_dim = 8
        action_dim = 4
        model = FullyConnectedParametricDQN(
            state_dim,
            action_dim,
            sizes=[8, 4],
            activations=["relu", "relu"],
            use_batch_norm=True,
        )
        input = model.input_prototype()
        self.assertEqual((1, state_dim), input.state.float_features.shape)
        self.assertEqual((1, action_dim), input.action.float_features.shape)
        # Using batch norm requires more than 1 example in training, avoid that
        model.eval()
        single_q_value = model(input)
        self.assertEqual((1, 1), single_q_value.q_value.shape)

    def test_save_load(self):
        state_dim = 8
        action_dim = 4
        model = FullyConnectedParametricDQN(
            state_dim,
            action_dim,
            sizes=[8, 4],
            activations=["relu", "relu"],
            use_batch_norm=False,
        )
        expected_num_params, expected_num_inputs, expected_num_outputs = 6, 2, 1
        check_save_load(
            self, model, expected_num_params, expected_num_inputs, expected_num_outputs
        )

    def test_save_load_batch_norm(self):
        state_dim = 8
        action_dim = 4
        model = FullyConnectedParametricDQN(
            state_dim,
            action_dim,
            sizes=[8, 4],
            activations=["relu", "relu"],
            use_batch_norm=True,
        )
        # Freezing batch_norm
        model.eval()
        expected_num_params, expected_num_inputs, expected_num_outputs = 21, 2, 1
        check_save_load(
            self, model, expected_num_params, expected_num_inputs, expected_num_outputs
        )
