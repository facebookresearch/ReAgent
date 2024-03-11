#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import unittest

import torch
from reagent.core.parameters import NormalizationData, NormalizationParameters
from reagent.core.types import FeatureData
from reagent.net_builder import value
from reagent.net_builder.unions import ValueNetBuilder__Union
from reagent.preprocessing.identify_types import CONTINUOUS


class TestValueNetBuilder(unittest.TestCase):
    def test_fully_connected(self) -> None:
        # pyre-fixme[28]: Unexpected keyword argument `FullyConnected`.
        chooser = ValueNetBuilder__Union(
            FullyConnected=value.fully_connected.FullyConnected()
        )
        builder = chooser.value
        state_dim = 3
        normalization_data = NormalizationData(
            dense_normalization_parameters={
                i: NormalizationParameters(feature_type=CONTINUOUS)
                for i in range(state_dim)
            }
        )
        value_network = builder.build_value_network(normalization_data)
        batch_size = 5
        x = FeatureData(float_features=torch.randn(batch_size, state_dim))
        y = value_network(x)
        self.assertEqual(y.shape, (batch_size, 1))
