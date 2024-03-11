#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging
import unittest
from typing import Union

import torch
import torch.nn as nn
from reagent.core import types as rlt

from reagent.models.dueling_q_network import DuelingQNetwork, ParametricDuelingQNetwork
from reagent.test.models.test_utils import run_model_jit_trace

logger = logging.getLogger(__name__)


class DuelingQNetworkTorchScriptWrapper(nn.Module):
    def __init__(
        self,
        model: Union[
            DuelingQNetwork,
            ParametricDuelingQNetwork,
        ],
    ):
        super().__init__()
        self.model = model

    def forward(self, state_float_features: torch.Tensor):
        return self.model(rlt.FeatureData(float_features=state_float_features))


class ParametricDuelingQNetworkTorchScriptWrapper(nn.Module):
    def __init__(
        self,
        model: Union[
            DuelingQNetwork,
            ParametricDuelingQNetwork,
        ],
    ):
        super().__init__()
        self.model = model

    def forward(
        self, state_float_features: torch.Tensor, action_float_features: torch.Tensor
    ):
        return self.model(
            rlt.FeatureData(float_features=state_float_features),
            rlt.FeatureData(float_features=action_float_features),
        )


class TestDuelingQNetwork(unittest.TestCase):
    def check_save_load(self, model: Union[DuelingQNetwork, ParametricDuelingQNetwork]):
        if isinstance(model, ParametricDuelingQNetwork):
            script_model = ParametricDuelingQNetworkTorchScriptWrapper(model)
        else:
            script_model = DuelingQNetworkTorchScriptWrapper(model)
        run_model_jit_trace(model, script_model)

    def test_discrete_action(self):
        state_dim = 8
        action_dim = 4
        model = DuelingQNetwork.make_fully_connected(
            state_dim,
            action_dim,
            layers=[8, 4],
            activations=["relu", "relu"],
            use_batch_norm=True,
        )
        input = model.input_prototype()
        self.assertEqual((1, state_dim), input.float_features.shape)
        # Using batch norm requires more than 1 example in training, avoid that
        model.eval()
        q_values = model(input)
        self.assertEqual((1, action_dim), q_values.shape)

    def test_parametric_action(self):
        state_dim = 8
        action_dim = 4
        model = ParametricDuelingQNetwork.make_fully_connected(
            state_dim, action_dim, [8, 4], ["relu", "relu"], use_batch_norm=True
        )
        state, action = model.input_prototype()
        self.assertEqual((1, state_dim), state.float_features.shape)
        self.assertEqual((1, action_dim), action.float_features.shape)
        # Using batch norm requires more than 1 example in training, avoid that
        model.eval()
        q_values = model(state, action)
        self.assertEqual((1, 1), q_values.shape)

    def test_save_load_discrete_action(self):
        state_dim = 8
        action_dim = 4
        model = DuelingQNetwork.make_fully_connected(
            state_dim, action_dim, layers=[8, 4], activations=["relu", "relu"]
        )
        self.check_save_load(model)

    def test_save_load_parametric_action(self):
        state_dim = 8
        action_dim = 4
        model = ParametricDuelingQNetwork.make_fully_connected(
            state_dim, action_dim, [8, 4], ["relu", "relu"]
        )
        self.check_save_load(model)

    def test_save_load_discrete_action_batch_norm(self):
        state_dim = 8
        action_dim = 4
        model = DuelingQNetwork.make_fully_connected(
            state_dim,
            action_dim,
            layers=[8, 4],
            activations=["relu", "relu"],
            use_batch_norm=True,
        )
        # Freezing batch_norm
        model.eval()
        self.check_save_load(model)

    def test_save_load_parametric_action_batch_norm(self):
        state_dim = 8
        action_dim = 4
        model = ParametricDuelingQNetwork.make_fully_connected(
            state_dim, action_dim, [8, 4], ["relu", "relu"], use_batch_norm=True
        )
        # Freezing batch_norm
        model.eval()
        self.check_save_load(model)
