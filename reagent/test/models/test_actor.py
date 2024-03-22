#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging
import unittest

from typing import Union

# pyre-fixme[21]: Could not find module `numpy.testing`.
import numpy.testing as npt
import torch
import torch.nn as nn
from reagent.core import types as rlt
from reagent.models.actor import (
    DirichletFullyConnectedActor,
    FullyConnectedActor,
    GaussianFullyConnectedActor,
)
from reagent.test.models.test_utils import run_model_jit_trace

logger = logging.getLogger(__name__)


class ActorTorchScriptWrapper(nn.Module):
    def __init__(
        self,
        model: Union[
            FullyConnectedActor,
            GaussianFullyConnectedActor,
            DirichletFullyConnectedActor,
        ],
    ):
        super().__init__()
        self.model = model

    def forward(self, state_float_features: torch.Tensor):
        actor_output = self.model(rlt.FeatureData(float_features=state_float_features))
        return actor_output.action, actor_output.log_prob


class TestActorBase(unittest.TestCase):
    def check_save_load(
        self,
        model: Union[
            FullyConnectedActor,
            GaussianFullyConnectedActor,
            DirichletFullyConnectedActor,
        ],
        stochastic: bool,
    ):
        # jit.trace can't trace models with stochastic output
        if stochastic:
            return

        script_model = ActorTorchScriptWrapper(model)

        def compare_func(model_output, script_model_output):
            action, log_prob = script_model_output
            assert torch.all(action == model_output.action)
            assert torch.all(log_prob == model_output.log_prob)

        run_model_jit_trace(model, script_model, compare_func)


class TestFullyConnectedActor(TestActorBase):
    def test_basic(self):
        state_dim = 8
        action_dim = 4
        model = FullyConnectedActor(
            state_dim,
            action_dim,
            sizes=[7, 6],
            activations=["relu", "relu"],
            use_batch_norm=True,
        )
        input = model.input_prototype()
        self.assertEqual((1, state_dim), input.float_features.shape)
        # Using batch norm requires more than 1 example in training, avoid that
        model.eval()
        action = model(input)
        self.assertEqual((1, action_dim), action.action.shape)
        self.assertEqual((1, 1), action.log_prob.shape)

    def test_save_load(self):
        state_dim = 8
        action_dim = 4
        model = FullyConnectedActor(
            state_dim,
            action_dim,
            sizes=[7, 6],
            activations=["relu", "relu"],
            use_batch_norm=False,
        )
        self.check_save_load(model, stochastic=False)

    def test_save_load_batch_norm(self):
        state_dim = 8
        action_dim = 4
        model = FullyConnectedActor(
            state_dim,
            action_dim,
            sizes=[7, 6],
            activations=["relu", "relu"],
            use_batch_norm=True,
        )
        # Freezing batch_norm
        model.eval()
        self.check_save_load(model, stochastic=False)


class TestGaussianFullyConnectedActor(TestActorBase):
    def test_basic(self):
        state_dim = 8
        action_dim = 4
        model = GaussianFullyConnectedActor(
            state_dim,
            action_dim,
            sizes=[7, 6],
            activations=["relu", "relu"],
            use_batch_norm=True,
        )
        input = model.input_prototype()
        self.assertEqual((1, state_dim), input.float_features.shape)
        # Using batch norm requires more than 1 example in training, avoid that
        model.eval()
        action = model(input)
        self.assertEqual((1, action_dim), action.action.shape)
        self.assertEqual((1, 1), action.log_prob.shape)

    def test_save_load(self):
        state_dim = 8
        action_dim = 4
        model = GaussianFullyConnectedActor(
            state_dim,
            action_dim,
            sizes=[7, 6],
            activations=["relu", "relu"],
            use_batch_norm=False,
        )
        self.check_save_load(model, stochastic=True)

    def test_save_load_batch_norm(self):
        state_dim = 8
        action_dim = 4
        model = GaussianFullyConnectedActor(
            state_dim,
            action_dim,
            sizes=[7, 6],
            activations=["relu", "relu"],
            use_batch_norm=True,
        )
        # Freezing batch_norm
        model.eval()
        self.check_save_load(model, stochastic=True)

    def test_get_log_prob(self):
        torch.manual_seed(0)
        state_dim = 8
        action_dim = 4
        model = GaussianFullyConnectedActor(
            state_dim,
            action_dim,
            sizes=[7, 6],
            activations=["relu", "relu"],
            use_batch_norm=False,
        )
        input = model.input_prototype()
        self.assertEqual((1, state_dim), input.float_features.shape)
        action = model(input)
        squashed_action = action.action.detach()
        action_log_prob = model.get_log_prob(input, squashed_action).detach()
        npt.assert_allclose(action.log_prob.detach(), action_log_prob, rtol=1e-4)


class TestDirichletFullyConnectedActor(TestActorBase):
    def test_basic(self):
        state_dim = 8
        action_dim = 4
        model = DirichletFullyConnectedActor(
            state_dim,
            action_dim,
            sizes=[7, 6],
            activations=["relu", "relu"],
            use_batch_norm=True,
        )
        input = model.input_prototype()
        self.assertEqual((1, state_dim), input.float_features.shape)
        # Using batch norm requires more than 1 example in training, avoid that
        model.eval()
        action = model(input)
        self.assertEqual((1, action_dim), action.action.shape)
        self.assertEqual((1, 1), action.log_prob.shape)

    def test_save_load(self):
        state_dim = 8
        action_dim = 4
        model = DirichletFullyConnectedActor(
            state_dim,
            action_dim,
            sizes=[7, 6],
            activations=["relu", "relu"],
            use_batch_norm=False,
        )
        self.check_save_load(model, stochastic=True)

    def test_save_load_batch_norm(self):
        state_dim = 8
        action_dim = 4
        model = DirichletFullyConnectedActor(
            state_dim,
            action_dim,
            sizes=[7, 6],
            activations=["relu", "relu"],
            use_batch_norm=True,
        )
        # Freezing batch_norm
        model.eval()
        self.check_save_load(model, stochastic=True)
