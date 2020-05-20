#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import torch
from reagent.net_builder import value
from reagent.net_builder.unions import ValueNetBuilder__Union
from reagent.parameters import NormalizationParameters
from reagent.preprocessing.identify_types import CONTINUOUS


class TestValueNetBuilder(unittest.TestCase):
    def test_fully_connected(self):
        chooser = ValueNetBuilder__Union(
            FullyConnected=value.fully_connected.FullyConnected()
        )
        builder = chooser.value
        state_dim = 3
        norm_params = {
            i: NormalizationParameters(feature_type=CONTINUOUS)
            for i in range(state_dim)
        }
        value_network = builder.build_value_network(norm_params)
        batch_size = 5
        x = torch.randn(batch_size, state_dim)
        y = value_network(x)
        self.assertEqual(y.shape, (batch_size, 1))
