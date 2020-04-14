#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

from ml.rl.preprocessing import identify_types
from ml.rl.test.preprocessing.preprocessing_util import (
    BINARY_FEATURE_ID,
    BOXCOX_FEATURE_ID,
    CONTINUOUS_FEATURE_ID,
    ENUM_FEATURE_ID,
    PROBABILITY_FEATURE_ID,
    QUANTILE_FEATURE_ID,
    read_data,
)


class TestTypeIdentification(unittest.TestCase):
    def test_identification(self):
        feature_value_map = read_data()

        types = {}
        for name, values in feature_value_map.items():
            types[name] = identify_types.identify_type(values)

        # Examples through manual inspection
        self.assertEqual(types[BINARY_FEATURE_ID], identify_types.BINARY)
        self.assertEqual(types[CONTINUOUS_FEATURE_ID], identify_types.CONTINUOUS)

        # We don't yet know the boxcox type
        self.assertEqual(types[BOXCOX_FEATURE_ID], identify_types.CONTINUOUS)

        # We don't yet know the quantile type
        self.assertEqual(types[QUANTILE_FEATURE_ID], identify_types.CONTINUOUS)
        self.assertEqual(types[ENUM_FEATURE_ID], identify_types.ENUM)
        self.assertEqual(types[PROBABILITY_FEATURE_ID], identify_types.PROBABILITY)
