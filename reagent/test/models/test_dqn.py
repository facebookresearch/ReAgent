#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging
import unittest

import torch
import torch.nn as nn
from reagent.core import types as rlt
from reagent.models.dqn import FullyConnectedDQN
from reagent.test.models.test_utils import run_model_jit_trace

logger = logging.getLogger(__name__)


class FullyConnectedDQNTorchScriptWrapper(nn.Module):
    def __init__(self, model: FullyConnectedDQN):
        super().__init__()
        self.model = model

    def forward(self, state_float_features: torch.Tensor):
        return self.model(
            rlt.FeatureData(float_features=state_float_features),
        )


class TestFullyConnectedDQN(unittest.TestCase):
    def check_save_load(self, model: FullyConnectedDQN):
        """
        Test if a model is torch.jit.tracable
        """
        script_model = FullyConnectedDQNTorchScriptWrapper(model)
        run_model_jit_trace(model, script_model)

    def test_basic(self):
        state_dim = 8
        action_dim = 4
        model = FullyConnectedDQN(
            state_dim,
            action_dim,
            sizes=[8, 4],
            activations=["relu", "relu"],
            use_batch_norm=True,
        )
        input = model.input_prototype()
        self.assertEqual((1, state_dim), input.float_features.shape)
        # Using batch norm requires more than 1 example in training, avoid that
        model.eval()
        q_values = model(input)
        self.assertEqual((1, action_dim), q_values.shape)

    def test_save_load(self):
        state_dim = 8
        action_dim = 4
        model = FullyConnectedDQN(
            state_dim,
            action_dim,
            sizes=[8, 4],
            activations=["relu", "relu"],
            use_batch_norm=False,
        )
        self.check_save_load(model)

    def test_save_load_batch_norm(self):
        state_dim = 8
        action_dim = 4
        model = FullyConnectedDQN(
            state_dim,
            action_dim,
            sizes=[8, 4],
            activations=["relu", "relu"],
            use_batch_norm=True,
        )
        # Freezing batch_norm
        model.eval()
        self.check_save_load(model)
