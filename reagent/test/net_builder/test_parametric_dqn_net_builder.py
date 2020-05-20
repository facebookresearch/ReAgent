#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

from reagent.net_builder import parametric_dqn
from reagent.net_builder.unions import ParametricDQNNetBuilder__Union
from reagent.parameters import NormalizationData, NormalizationParameters
from reagent.preprocessing.identify_types import CONTINUOUS


try:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbParametricDqnPredictorWrapper as ParametricDqnPredictorWrapper,
    )
except ImportError:
    from reagent.prediction.predictor_wrapper import ParametricDqnPredictorWrapper


class TestParametricDQNNetBuilder(unittest.TestCase):
    def _test_parametric_dqn_net_builder(
        self, chooser: ParametricDQNNetBuilder__Union
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
                    feature_type=CONTINUOUS, mean=0.0, stddev=1.0
                )
                for i in range(action_dim)
            }
        )
        q_network = builder.build_q_network(
            state_normalization_data, action_normalization_data
        )
        x = q_network.input_prototype()
        y = q_network(*x)
        self.assertEqual(y.shape, (1, 1))
        serving_module = builder.build_serving_module(
            q_network, state_normalization_data, action_normalization_data
        )
        self.assertIsInstance(serving_module, ParametricDqnPredictorWrapper)

    def test_fully_connected(self):
        # Intentionally used this long path to make sure we included it in __init__.py
        chooser = ParametricDQNNetBuilder__Union(
            FullyConnected=parametric_dqn.fully_connected.FullyConnected()
        )
        self._test_parametric_dqn_net_builder(chooser)
