#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import numpy as np
import numpy.testing as npt
import six
import torch
from caffe2.python import core
from ml.rl.caffe_utils import PytorchCaffe2Converter
from ml.rl.preprocessing import identify_types, normalization
from ml.rl.preprocessing.identify_types import BOXCOX, CONTINUOUS, ENUM
from ml.rl.preprocessing.normalization import (
    MISSING_VALUE,
    NormalizationParameters,
    sort_features_by_normalization,
)
from ml.rl.preprocessing.preprocessor import Preprocessor
from ml.rl.test.base.utils import NumpyFeatureProcessor
from ml.rl.test.preprocessing.preprocessing_util import (
    BOXCOX_FEATURE_ID,
    CONTINUOUS_ACTION_FEATURE_ID,
    CONTINUOUS_ACTION_FEATURE_ID_2,
    ENUM_FEATURE_ID,
    PROBABILITY_FEATURE_ID,
    id_to_type,
    read_data,
)
from scipy import special


class TestPreprocessing(unittest.TestCase):
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

        preprocessor = Preprocessor(normalization_parameters, False)
        sorted_features, _ = sort_features_by_normalization(normalization_parameters)
        input_matrix = np.zeros([10000, len(sorted_features)], dtype=np.float32)
        for i, feature in enumerate(sorted_features):
            input_matrix[:, i] = feature_value_map[feature]
        normalized_feature_matrix = preprocessor.forward(input_matrix)

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
            v = v.numpy()
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
                    if abs(original_feature - MISSING_VALUE) < 0.01:
                        self.assertEqual(0.0, np.sum(row))
                    else:
                        self.assertEqual(
                            possible_value_map[original_feature],
                            np.where(row == 1)[0][0],
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
        preprocessor = Preprocessor(normalization_parameters, False)

        inputs = np.zeros([4, 3], dtype=np.float32)
        feature_ids = [2, 1, 3]  # Sorted according to feature type
        inputs[:, feature_ids.index(1)] = [12, 4, 2, 2]
        inputs[:, feature_ids.index(2)] = [1.0, 2.0, 3.0, 3.0]
        inputs[:, feature_ids.index(3)] = [15, 3, 15, normalization.MISSING_VALUE]
        normalized_feature_matrix = preprocessor.forward(inputs)

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
            values[0] = MISSING_VALUE  # Set one entry to MISSING_VALUE to test that

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

    def test_preprocessing_network_onnx(self):
        feature_value_map = read_data()

        for feature_name, feature_values in feature_value_map.items():
            normalization_parameters = normalization.identify_parameter(
                feature_name,
                feature_values,
                feature_type=self._feature_type_override(feature_name),
            )
            feature_values[
                0
            ] = MISSING_VALUE  # Set one entry to MISSING_VALUE to test that
            feature_values_matrix = np.expand_dims(feature_values, -1)

            preprocessor = Preprocessor({feature_name: normalization_parameters}, False)
            normalized_feature_values = preprocessor.forward(feature_values_matrix)

            input_blob, output_blob, netdef = PytorchCaffe2Converter.pytorch_net_to_caffe2_netdef(
                preprocessor, 1, False, float_input=True
            )

            preproc_workspace = netdef.workspace

            preproc_workspace.FeedBlob(input_blob, feature_values_matrix)

            preproc_workspace.RunNetOnce(core.Net(netdef.init_net))
            preproc_workspace.RunNetOnce(core.Net(netdef.predict_net))

            normalized_feature_values_onnx = netdef.workspace.FetchBlob(output_blob)
            tolerance = 0.0001
            non_matching = np.where(
                np.logical_not(
                    np.isclose(
                        normalized_feature_values,
                        normalized_feature_values_onnx,
                        rtol=tolerance,
                        atol=tolerance,
                    )
                )
            )
            self.assertTrue(
                np.all(
                    np.isclose(
                        normalized_feature_values,
                        normalized_feature_values_onnx,
                        rtol=tolerance,
                        atol=tolerance,
                    )
                ),
                "{} does not match: {} \n!=\n {}".format(
                    feature_name,
                    normalized_feature_values[non_matching].tolist()[0:10],
                    normalized_feature_values_onnx[non_matching].tolist()[0:10],
                ),
            )

    def test_quantile_boundary_logic(self):
        """Test quantile logic when feaure value == quantile boundary."""
        input = torch.tensor([[0.0], [80.0], [100.0]])
        norm_params = NormalizationParameters(
            feature_type="QUANTILE",
            boxcox_lambda=None,
            boxcox_shift=None,
            mean=0,
            stddev=1,
            possible_values=None,
            quantiles=[0.0, 80.0, 100.0],
            min_value=0.0,
            max_value=100.0,
        )
        preprocessor = Preprocessor({1: norm_params}, False)
        output = preprocessor._preprocess_QUANTILE(0, input.float(), [norm_params])

        expected_output = torch.tensor([[0.0], [0.5], [1.0]])

        self.assertTrue(np.all(np.isclose(output, expected_output)))

    def test_preprocessing_network(self):
        feature_value_map = read_data()

        normalization_parameters = {}
        name_preprocessed_blob_map = {}

        for feature_name, feature_values in feature_value_map.items():
            normalization_parameters[feature_name] = normalization.identify_parameter(
                feature_name,
                feature_values,
                feature_type=self._feature_type_override(feature_name),
            )
            feature_values[
                0
            ] = MISSING_VALUE  # Set one entry to MISSING_VALUE to test that

            preprocessor = Preprocessor(
                {feature_name: normalization_parameters[feature_name]}, False
            )
            feature_values_matrix = np.expand_dims(feature_values, -1)
            normalized_feature_values = preprocessor.forward(feature_values_matrix)
            name_preprocessed_blob_map[feature_name] = normalized_feature_values.numpy()

        test_features = NumpyFeatureProcessor.preprocess(
            feature_value_map, normalization_parameters
        )

        for feature_name in feature_value_map:
            normalized_features = name_preprocessed_blob_map[feature_name]
            if feature_name != ENUM_FEATURE_ID:
                normalized_features = np.squeeze(normalized_features, -1)

            tolerance = 0.01
            if feature_name == BOXCOX_FEATURE_ID:
                # At the limit, boxcox has some numerical instability
                tolerance = 0.5
            non_matching = np.where(
                np.logical_not(
                    np.isclose(
                        normalized_features.flatten(),
                        test_features[feature_name].flatten(),
                        rtol=tolerance,
                        atol=tolerance,
                    )
                )
            )
            self.assertTrue(
                np.all(
                    np.isclose(
                        normalized_features.flatten(),
                        test_features[feature_name].flatten(),
                        rtol=tolerance,
                        atol=tolerance,
                    )
                ),
                "{} does not match: {} \n!=\n {}".format(
                    feature_name,
                    normalized_features.flatten()[non_matching],
                    test_features[feature_name].flatten()[non_matching],
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
