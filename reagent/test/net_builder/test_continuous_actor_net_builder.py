#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

from reagent.net_builder import continuous_actor
from reagent.net_builder.unions import ContinuousActorNetBuilder__Union
from reagent.parameters import NormalizationData, NormalizationParameters
from reagent.preprocessing.identify_types import CONTINUOUS


try:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbActorPredictorWrapper as ActorPredictorWrapper,
    )
except ImportError:
    from reagent.prediction.predictor_wrapper import ActorPredictorWrapper


class TestContinuousActorNetBuilder(unittest.TestCase):
    def _test_actor_net_builder(
        self, chooser: ContinuousActorNetBuilder__Union
    ) -> None:
        builder = chooser.value
        state_dim = 3
        state_normalization_data = NormalizationData(
            dense_normalization_parameters={
                i: NormalizationParameters(
                    feature_type=CONTINUOUS, mean=0.0, stddev=1.0
                )
                for i in range(state_dim)
            }
        )
        action_dim = 2
        action_normalization_data = NormalizationData(
            dense_normalization_parameters={
                i: NormalizationParameters(
                    feature_type=builder.default_action_preprocessing,
                    min_value=0.0,
                    max_value=1.0,
                )
                for i in range(action_dim)
            }
        )
        actor_network = builder.build_actor(
            state_normalization_data, action_normalization_data
        )
        x = actor_network.input_prototype()
        y = actor_network(x)
        action = y.action
        log_prob = y.log_prob
        self.assertEqual(action.shape, (1, action_dim))
        self.assertEqual(log_prob.shape, (1, 1))
        serving_module = builder.build_serving_module(
            actor_network, state_normalization_data, action_normalization_data
        )
        self.assertIsInstance(serving_module, ActorPredictorWrapper)

    def test_gaussian_fully_connected(self):
        # Intentionally used this long path to make sure we included it in __init__.py
        chooser = ContinuousActorNetBuilder__Union(
            GaussianFullyConnected=continuous_actor.gaussian_fully_connected.GaussianFullyConnected()
        )
        self._test_actor_net_builder(chooser)

    def test_dirichlet_fully_connected(self):
        # Intentionally used this long path to make sure we included it in __init__.py
        chooser = ContinuousActorNetBuilder__Union(
            DirichletFullyConnected=continuous_actor.dirichlet_fully_connected.DirichletFullyConnected()
        )
        self._test_actor_net_builder(chooser)
