#!/usr/bin/env python3

import unittest

from ml.rl.preprocessing import identify_types
from ml.rl.test import preprocessing_util


class TestTypeIdentification(unittest.TestCase):
    def test_identification(self):
        _, feature_value_map = preprocessing_util.read_data()

        types = {}
        for name, values in feature_value_map.items():
            types[name] = identify_types.identify_type(values)

        # Examples through manual inspection
        self.assertEqual(types[identify_types.BINARY], identify_types.BINARY)
        self.assertEqual(types[identify_types.CONTINUOUS], identify_types.CONTINUOUS)

        # We don't yet know the boxcox type
        self.assertEqual(types[identify_types.BOXCOX], identify_types.CONTINUOUS)

        # We don't yet know the quantile type
        self.assertEqual(types[identify_types.QUANTILE], identify_types.CONTINUOUS)
        self.assertEqual(types[identify_types.ENUM], identify_types.ENUM)
        self.assertEqual(types[identify_types.PROBABILITY], identify_types.PROBABILITY)
