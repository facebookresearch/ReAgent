from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

from ml.rl.preprocessing import identify_types
from ml.rl.test import preprocessing_util


class TestTypeIdentification(unittest.TestCase):
    def test_feature_parsing(self):
        feature_value_map = preprocessing_util.read_data()

        # There are features and we have mapped all them all
        self.assertTrue(len(feature_value_map) > 3)

        # A few samples based on manual inspection
        self.assertFalse(
            (
                feature_value_map['413'] - [
                    473.763927022, 65.0, 65.0, 1.0, 50.0, 50.0, 2.0, 2.0, 1.0,
                    23.0
                ]
            ).any()
        )
        self.assertFalse(
            (
                feature_value_map['186'] -
                [0.0, 0.0, 0.0, 11.0, 1.0, 1.0, 7.0, 7.0, 14.0, 0.0]
            ).any()
        )

    def test_identification(self):
        feature_value_map = preprocessing_util.read_data()

        types = identify_types.identify_types(feature_value_map)

        # Examples through manual inspection
        self.assertEqual(types['179'], identify_types.BINARY)
        self.assertEqual(types['124'], identify_types.CONTINUOUS)
        self.assertEqual(types['74'], identify_types.PROBABILITY)
