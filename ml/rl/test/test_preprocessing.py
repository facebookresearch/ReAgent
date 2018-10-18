#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import numpy as np
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
from ml.rl.test.preprocessing_util import (
    BOXCOX_FEATURE_ID,
    ENUM_FEATURE_ID,
    PROBABILITY_FEATURE_ID,
    id_to_type,
    read_data,
)
from scipy import special, stats


class TestPreprocessing(unittest.TestCase):
    def _value_to_quantile(self, original_value, quantiles):
        if original_value <= quantiles[0]:
            return 0.0
        if original_value >= quantiles[-1]:
            return 1.0
        n_quantiles = float(len(quantiles) - 1)
        right = np.searchsorted(quantiles, original_value)
        left = right - 1
        interpolated = (
            left
            + (
                (original_value - quantiles[left])
                / ((quantiles[right] + 1e-6) - quantiles[left])
            )
        ) / n_quantiles
        return interpolated

    def test_prepare_normalization_and_normalize(self):
        feature_value_map = read_data()

        normalization_parameters = {}
        for name, values in feature_value_map.items():
            normalization_parameters[name] = normalization.identify_parameter(
                values, 10
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
        preprocessor.clamp = False
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
                    expected = self._value_to_quantile(
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
        preprocessor.clamp = False

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
            normalization_parameters[name] = normalization.identify_parameter(values)
            values[0] = MISSING_VALUE  # Set one entry to MISSING_VALUE to test that

        s = normalization.serialize(normalization_parameters)
        read_parameters = normalization.deserialize(s)
        self.assertEqual(read_parameters, normalization_parameters)

    def preprocess_feature(self, feature, parameters):
        is_not_empty = 1 - np.isclose(feature, normalization.MISSING_VALUE)
        if parameters.feature_type == identify_types.BINARY:
            # Binary features are always 1 unless they are 0
            return ((feature != 0) * is_not_empty).astype(np.float32)
        if parameters.boxcox_lambda is not None:
            feature = stats.boxcox(
                np.maximum(
                    feature + parameters.boxcox_shift, normalization.BOX_COX_MARGIN
                ),
                parameters.boxcox_lambda,
            )
        # No *= to ensure consistent out-of-place operation.
        if parameters.feature_type == identify_types.PROBABILITY:
            feature = special.logit(np.clip(feature, 1e-6, 1.0))
        elif parameters.feature_type == identify_types.QUANTILE:
            transformed_feature = np.zeros_like(feature)
            for i in six.moves.range(feature.shape[0]):
                transformed_feature[i] = self._value_to_quantile(
                    feature[i], parameters.quantiles
                )
            feature = transformed_feature
        elif parameters.feature_type == identify_types.ENUM:
            possible_values = parameters.possible_values
            mapping = {}
            for i, possible_value in enumerate(possible_values):
                mapping[possible_value] = i
            output_feature = np.zeros((len(feature), len(possible_values)))
            for i, val in enumerate(feature):
                if abs(val - MISSING_VALUE) < 1e-2:
                    continue
                output_feature[i][mapping[val]] = 1.0
            return output_feature
        else:
            feature = feature - parameters.mean
            feature /= parameters.stddev
        feature *= is_not_empty
        return feature

    def preprocess(self, features, parameters):
        result = {}
        for feature_name in features:
            result[feature_name] = self.preprocess_feature(
                features[feature_name], parameters[feature_name]
            )
        return result

    def test_preprocessing_network_onnx(self):
        feature_value_map = read_data()

        for feature_name, feature_values in feature_value_map.items():
            normalization_parameters = normalization.identify_parameter(feature_values)
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
                feature_values
            )
            feature_values[
                0
            ] = MISSING_VALUE  # Set one entry to MISSING_VALUE to test that

            preprocessor = Preprocessor(
                {feature_name: normalization_parameters[feature_name]}, False
            )
            preprocessor.clamp = False
            feature_values_matrix = np.expand_dims(feature_values, -1)
            normalized_feature_values = preprocessor.forward(feature_values_matrix)
            name_preprocessed_blob_map[feature_name] = normalized_feature_values.numpy()

        test_features = self.preprocess(feature_value_map, normalization_parameters)

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
            probability_values, feature_type=identify_types.BINARY
        )
        self.assertEqual(parameter.feature_type, "BINARY")
