#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import numpy as np
import numpy.testing as npt
import six
from caffe2.python import core, workspace
from ml.rl.caffe_utils import C2
from ml.rl.preprocessing import identify_types, normalization
from ml.rl.preprocessing.identify_types import BOXCOX, CONTINUOUS, ENUM
from ml.rl.preprocessing.normalization import (
    NormalizationParameters,
    sort_features_by_normalization,
)
from ml.rl.preprocessing.preprocessor_net import PreprocessorNet
from ml.rl.test.base.utils import NumpyFeatureProcessor
from ml.rl.test.preprocessing.preprocessing_util import (
    BOXCOX_FEATURE_ID,
    ENUM_FEATURE_ID,
    PROBABILITY_FEATURE_ID,
    id_to_type,
    read_data,
)
from scipy import special


class TestNormalization(unittest.TestCase):
    def _feature_type_override(self, feature_id):
        """
        This should only be used to test CONTINUOUS_ACTION
        """
        if id_to_type(feature_id) == identify_types.CONTINUOUS_ACTION:
            return identify_types.CONTINUOUS_ACTION
        return None

    def test_prepare_normalization_and_normalize(self):
        feature_value_map = read_data()

        normalization_parameters = {}
        for name, values in feature_value_map.items():
            normalization_parameters[name] = normalization.identify_parameter(
                name, values, 10, feature_type=self._feature_type_override(name)
            )
        for k, v in normalization_parameters.items():
            if id_to_type(k) == CONTINUOUS:
                self.assertEqual(v.feature_type, CONTINUOUS)
                self.assertIs(v.boxcox_lambda, None)
                self.assertIs(v.boxcox_shift, None)
            elif id_to_type(k) == BOXCOX:
                self.assertEqual(v.feature_type, BOXCOX)
                self.assertIsNot(v.boxcox_lambda, None)
                self.assertIsNot(v.boxcox_shift, None)
            else:
                assert v.feature_type == id_to_type(k)
        sorted_features, _ = sort_features_by_normalization(normalization_parameters)

        norm_net = core.Net("net")
        C2.set_net(norm_net)
        preprocessor = PreprocessorNet()
        input_matrix = np.zeros([10000, len(sorted_features)], dtype=np.float32)
        for i, feature in enumerate(sorted_features):
            input_matrix[:, i] = feature_value_map[feature]
        input_matrix_blob = "input_matrix_blob"
        workspace.FeedBlob(input_matrix_blob, np.array([], dtype=np.float32))
        output_blob, _ = preprocessor.normalize_dense_matrix(
            input_matrix_blob, sorted_features, normalization_parameters, "", False
        )
        workspace.FeedBlob(input_matrix_blob, input_matrix)
        workspace.RunNetOnce(norm_net)
        normalized_feature_matrix = workspace.FetchBlob(output_blob)

        normalized_features = {}
        on_column = 0
        for feature in sorted_features:
            norm = normalization_parameters[feature]
            if norm.feature_type == ENUM:
                column_size = len(norm.possible_values)
            else:
                column_size = 1
            normalized_features[feature] = normalized_feature_matrix[
                :, on_column : (on_column + column_size)
            ]
            on_column += column_size

        self.assertTrue(
            all(
                [
                    np.isfinite(parameter.stddev) and np.isfinite(parameter.mean)
                    for parameter in normalization_parameters.values()
                ]
            )
        )
        for k, v in six.iteritems(normalized_features):
            self.assertTrue(np.all(np.isfinite(v)))
            feature_type = normalization_parameters[k].feature_type
            if feature_type == identify_types.PROBABILITY:
                sigmoidv = special.expit(v)
                self.assertTrue(
                    np.all(
                        np.logical_and(np.greater(sigmoidv, 0), np.less(sigmoidv, 1))
                    )
                )
            elif feature_type == identify_types.ENUM:
                possible_values = normalization_parameters[k].possible_values
                self.assertEqual(v.shape[0], len(feature_value_map[k]))
                self.assertEqual(v.shape[1], len(possible_values))

                possible_value_map = {}
                for i, possible_value in enumerate(possible_values):
                    possible_value_map[possible_value] = i

                for i, row in enumerate(v):
                    original_feature = feature_value_map[k][i]
                    self.assertEqual(
                        possible_value_map[original_feature], np.where(row == 1)[0][0]
                    )
            elif feature_type == identify_types.QUANTILE:
                for i, feature in enumerate(v[0]):
                    original_feature = feature_value_map[k][i]
                    expected = NumpyFeatureProcessor.value_to_quantile(
                        original_feature, normalization_parameters[k].quantiles
                    )
                    self.assertAlmostEqual(feature, expected, 2)
            elif feature_type == identify_types.BINARY:
                pass
            elif (
                feature_type == identify_types.CONTINUOUS
                or feature_type == identify_types.BOXCOX
            ):
                one_stddev = np.isclose(np.std(v, ddof=1), 1, atol=0.01)
                zero_stddev = np.isclose(np.std(v, ddof=1), 0, atol=0.01)
                zero_mean = np.isclose(np.mean(v), 0, atol=0.01)
                self.assertTrue(
                    np.all(zero_mean),
                    "mean of feature {} is {}, not 0".format(k, np.mean(v)),
                )
                self.assertTrue(np.all(np.logical_or(one_stddev, zero_stddev)))
            elif feature_type == identify_types.CONTINUOUS_ACTION:
                less_than_max = v < 1
                more_than_min = v > -1
                self.assertTrue(
                    np.all(less_than_max),
                    "values are not less than 1: {}".format(v[less_than_max == False]),
                )
                self.assertTrue(
                    np.all(more_than_min),
                    "values are not more than -1: {}".format(v[more_than_min == False]),
                )
            else:
                raise NotImplementedError()

    def test_normalize_dense_matrix_enum(self):
        normalization_parameters = {
            1: NormalizationParameters(
                identify_types.ENUM,
                None,
                None,
                None,
                None,
                [12, 4, 2],
                None,
                None,
                None,
            ),
            2: NormalizationParameters(
                identify_types.CONTINUOUS, None, 0, 0, 1, None, None, None, None
            ),
            3: NormalizationParameters(
                identify_types.ENUM, None, None, None, None, [15, 3], None, None, None
            ),
        }
        norm_net = core.Net("net")
        C2.set_net(norm_net)
        preprocessor = PreprocessorNet()

        inputs = np.zeros([4, 3], dtype=np.float32)
        feature_ids = [2, 1, 3]  # Sorted according to feature type
        inputs[:, feature_ids.index(1)] = [12, 4, 2, 2]
        inputs[:, feature_ids.index(2)] = [1.0, 2.0, 3.0, 3.0]
        inputs[:, feature_ids.index(3)] = [15, 3, 15, normalization.MISSING_VALUE]
        input_blob = C2.NextBlob("input_blob")
        workspace.FeedBlob(input_blob, np.array([0], dtype=np.float32))
        normalized_output_blob, _ = preprocessor.normalize_dense_matrix(
            input_blob, feature_ids, normalization_parameters, "", False
        )
        workspace.FeedBlob(input_blob, inputs)
        workspace.RunNetOnce(norm_net)
        normalized_feature_matrix = workspace.FetchBlob(normalized_output_blob)

        np.testing.assert_allclose(
            np.array(
                [
                    [1.0, 1, 0, 0, 1, 0],
                    [2.0, 0, 1, 0, 0, 1],
                    [3.0, 0, 0, 1, 1, 0],
                    [3.0, 0, 0, 1, 0, 0],  # Missing values should go to all 0
                ]
            ),
            normalized_feature_matrix,
        )

    def test_persistency(self):
        feature_value_map = read_data()
        normalization_parameters = {}
        for name, values in feature_value_map.items():
            normalization_parameters[name] = normalization.identify_parameter(
                name, values, feature_type=self._feature_type_override(name)
            )

        s = normalization.serialize(normalization_parameters)
        read_parameters = normalization.deserialize(s)
        # Unfortunately, Thrift serializatin seems to lose a bit of precision.
        # Using `==` will be false.
        self.assertEqual(read_parameters.keys(), normalization_parameters.keys())
        for k in normalization_parameters:
            self.assertEqual(
                read_parameters[k].feature_type,
                normalization_parameters[k].feature_type,
            )
            self.assertEqual(
                read_parameters[k].possible_values,
                normalization_parameters[k].possible_values,
            )
            for field in [
                "boxcox_lambda",
                "boxcox_shift",
                "mean",
                "stddev",
                "quantiles",
                "min_value",
                "max_value",
            ]:
                if getattr(normalization_parameters[k], field) is None:
                    self.assertEqual(
                        getattr(read_parameters[k], field),
                        getattr(normalization_parameters[k], field),
                    )
                else:
                    npt.assert_allclose(
                        getattr(read_parameters[k], field),
                        getattr(normalization_parameters[k], field),
                    )

    def test_preprocessing_network(self):
        feature_value_map = read_data()

        normalization_parameters = {}
        for name, values in feature_value_map.items():
            normalization_parameters[name] = normalization.identify_parameter(
                name, values, feature_type=self._feature_type_override(name)
            )
        test_features = NumpyFeatureProcessor.preprocess(
            feature_value_map, normalization_parameters
        )

        net = core.Net("PreprocessingTestNet")
        C2.set_net(net)
        preprocessor = PreprocessorNet()
        name_preprocessed_blob_map = {}
        for feature_name in feature_value_map:
            workspace.FeedBlob(str(feature_name), np.array([0], dtype=np.int32))
            preprocessed_blob, _ = preprocessor.preprocess_blob(
                str(feature_name), [normalization_parameters[feature_name]]
            )
            name_preprocessed_blob_map[feature_name] = preprocessed_blob

        workspace.CreateNet(net)

        for feature_name, feature_value in six.iteritems(feature_value_map):
            feature_value = np.expand_dims(feature_value, -1)
            workspace.FeedBlob(str(feature_name), feature_value)
        workspace.RunNetOnce(net)

        for feature_name in feature_value_map:
            normalized_features = workspace.FetchBlob(
                name_preprocessed_blob_map[feature_name]
            )
            if feature_name != ENUM_FEATURE_ID:
                normalized_features = np.squeeze(normalized_features, -1)

            tolerance = 0.01
            if feature_name == BOXCOX_FEATURE_ID:
                # At the limit, boxcox has some numerical instability
                tolerance = 0.5
            non_matching = np.where(
                np.logical_not(
                    np.isclose(
                        normalized_features,
                        test_features[feature_name],
                        rtol=tolerance,
                        atol=tolerance,
                    )
                )
            )
            self.assertTrue(
                np.all(
                    np.isclose(
                        normalized_features,
                        test_features[feature_name],
                        rtol=tolerance,
                        atol=tolerance,
                    )
                ),
                "{} does not match: {} {}".format(
                    feature_name,
                    normalized_features[non_matching].tolist(),
                    test_features[feature_name][non_matching].tolist(),
                ),
            )

    def test_type_override(self):
        # Take a feature that should be identified as probability
        feature_value_map = read_data()
        probability_values = feature_value_map[PROBABILITY_FEATURE_ID]

        # And ask for a binary anyways
        parameter = normalization.identify_parameter(
            "_", probability_values, feature_type=identify_types.BINARY
        )
        self.assertEqual(parameter.feature_type, "BINARY")
