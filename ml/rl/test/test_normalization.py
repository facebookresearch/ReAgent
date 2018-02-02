from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from scipy import special, stats
import numpy as np
import six
import unittest

from caffe2.python import core, workspace

from ml.rl.preprocessing import identify_types
from ml.rl.preprocessing import normalization
from ml.rl.preprocessing.preprocessor_net import prepare_normalization,\
    PreprocessorNet, normalize_feature_map, normalize_dense_matrix
from ml.rl.test import preprocessing_util
from ml.rl.preprocessing.normalization import NormalizationParameters


class TestNormalization(unittest.TestCase):
    def test_prepare_normalization_and_normalize(self):
        feature_value_map = preprocessing_util.read_data()

        normalization_parameters = {}
        for name, values in feature_value_map.items():
            normalization_parameters[name] = normalization.identify_parameter(
                values, 10
            )
        for k, v in normalization_parameters.items():
            if k == 'normal':
                self.assertEqual(v.feature_type, 'CONTINUOUS')
                self.assertIs(v.boxcox_lambda, None)
                self.assertIs(v.boxcox_shift, None)
            elif k == 'boxcox':
                self.assertEqual(v.feature_type, 'CONTINUOUS')
                self.assertIsNot(v.boxcox_lambda, None)
                self.assertIsNot(v.boxcox_shift, None)
            else:
                self.assertEqual(v.feature_type, k)

        features = list(feature_value_map.keys())
        norm_net = core.Net("net")
        blobname_template = '{}_blob'
        blob_map = prepare_normalization(
            norm_net, normalization_parameters, features, blobname_template,
            False
        )

        normalized_features = normalize_feature_map(
            feature_value_map, norm_net, features, blob_map, blobname_template
        )

        self.assertTrue(
            all(
                [
                    np.isfinite(parameter.stddev) and
                    np.isfinite(parameter.mean)
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
                        np.logical_and(
                            np.greater(sigmoidv, 0), np.less(sigmoidv, 1)
                        )
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
                        possible_value_map[original_feature],
                        np.where(row == 1)[0][0]
                    )
            elif feature_type == identify_types.QUANTILE:
                quantiles = normalization_parameters[k].quantiles
                for i, feature in enumerate(v[0]):
                    original_feature = feature_value_map[k][i]
                    count = 0
                    for quantile in quantiles:
                        if original_feature >= quantile:
                            count += 1
                    count /= float(len(quantiles))
                    self.assertAlmostEqual(feature, count, 2)
            elif feature_type == identify_types.BINARY:
                pass
            elif feature_type == identify_types.CONTINUOUS:
                one_stddev = np.isclose(np.std(v, ddof=1), 1, atol=0.01)
                zero_stddev = np.isclose(np.std(v, ddof=1), 0, atol=0.01)
                zero_mean = np.isclose(np.mean(v), 0, atol=0.01)
                self.assertTrue(
                    np.all(zero_mean),
                    'mean of feature {} is {}, not 0'.format(k, np.mean(v))
                )
                self.assertTrue(np.all(np.logical_or(one_stddev, zero_stddev)))
            else:
                raise NotImplementedError()

    def test_normalize_feature_map_enum(self):
        feature_name_1 = 'f1'
        feature_name_2 = 'f2'
        feature_name_3 = 'f3'
        normalization_parameters = {
            feature_name_1:
            NormalizationParameters(
                identify_types.ENUM, None, None, None, None, [12, 4, 2], None
            ),
            feature_name_2:
            NormalizationParameters(
                identify_types.CONTINUOUS, None, 0, 0, 1, None, None
            ),
            feature_name_3:
            NormalizationParameters(
                identify_types.ENUM, None, None, None, None, [15, 3], None
            )
        }

        feature_value_map = {
            feature_name_1:
            np.array([2, 4, 12, 12], dtype=np.float32),
            feature_name_2:
            np.array([1.9, 2.2, 5.0, 1.0], dtype=np.float32),
            feature_name_3:
            np.array([3, 3, 15, normalization.MISSING_VALUE], dtype=np.float32)
        }

        features = list(feature_value_map.keys())
        norm_net = core.Net("net")
        blobname_template = '{}_blob'
        blob_map = prepare_normalization(
            norm_net, normalization_parameters, features, blobname_template,
            False
        )
        normalized_features = normalize_feature_map(
            feature_value_map, norm_net, features, blob_map, blobname_template
        )

        for v in normalized_features.values():
            self.assertTrue(np.all(np.isfinite(v)))

        np.testing.assert_array_equal(
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]]),
            normalized_features[feature_name_1]
        )
        np.testing.assert_array_equal(
            np.array([[1.9, 2.2, 5.0, 1.0]], dtype=np.float32),
            normalized_features[feature_name_2]
        )
        np.testing.assert_array_equal(
            np.array(
                [
                    [0, 1],
                    [0, 1],
                    [1, 0],
                    [0, 0]  # Missing value should go to all 0
                ]
            ),
            normalized_features[feature_name_3]
        )

    def test_normalize_dense_matrix_enum(self):
        normalization_parameters = {
            'f1':
            NormalizationParameters(
                identify_types.ENUM, None, None, None, None, [12, 4, 2], None
            ),
            'f2':
            NormalizationParameters(
                identify_types.CONTINUOUS, None, 0, 0, 1, None, None
            ),
            'f3':
            NormalizationParameters(
                identify_types.ENUM, None, None, None, None, [15, 3], None
            )
        }
        features = list(normalization_parameters.keys())
        norm_net = core.Net("net")
        blobname_template = '{}_blob'
        blob_map = prepare_normalization(
            norm_net, normalization_parameters, features, blobname_template,
            False
        )

        inputs = np.array(
            [
                [12, 1.0, 15], [4, 2.0, 3], [2, 3.0, 15],
                [2, 3.0, normalization.MISSING_VALUE]
            ],
            dtype=np.float32
        )
        normalized_outputs = normalize_dense_matrix(
            inputs, features, normalization_parameters, blob_map, norm_net,
            blobname_template
        )

        np.testing.assert_array_equal(
            np.array(
                [
                    [1, 0, 0, 1.0, 1, 0],
                    [0, 1, 0, 2.0, 0, 1],
                    [0, 0, 1, 3.0, 1, 0],
                    [0, 0, 1, 3.0, 0, 0]  # Missing values should go to all 0
                ]
            ),
            normalized_outputs
        )

    def test_persistency(self):
        feature_value_map = preprocessing_util.read_data()
        normalization_parameters = {}
        for name, values in feature_value_map.items():
            normalization_parameters[name] = normalization.identify_parameter(
                values
            )

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
                    feature + parameters.boxcox_shift,
                    normalization.BOX_COX_MARGIN
                ), parameters.boxcox_lambda
            )
        # No *= to ensure consistent out-of-place operation.
        if parameters.feature_type == identify_types.PROBABILITY:
            feature = np.clip(feature, 0.01, 0.99)
            feature = special.logit(feature)
        elif parameters.feature_type == identify_types.QUANTILE:
            quantiles = parameters.quantiles
            values = np.zeros(feature.shape)
            for quantile in quantiles:
                values += feature >= quantile
            feature = values / float(len(quantiles))
        elif parameters.feature_type == identify_types.ENUM:
            possible_values = parameters.possible_values
            mapping = {}
            for i, possible_value in enumerate(possible_values):
                mapping[possible_value] = i
            output_feature = np.zeros((len(feature), len(possible_values)))
            for i, val in enumerate(feature):
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

    def test_preprocessing_network(self):
        feature_value_map = preprocessing_util.read_data()
        normalization_parameters = {}
        for name, values in feature_value_map.items():
            normalization_parameters[name] = normalization.identify_parameter(
                values
            )
        test_features = self.preprocess(
            feature_value_map, normalization_parameters
        )

        net = core.Net("PreprocessingTestNet")
        preprocessor = PreprocessorNet(net, False)
        for feature_name in feature_value_map:
            workspace.FeedBlob(feature_name, np.array([0], dtype=np.int32))
            preprocessor.preprocess_blob(
                feature_name, normalization_parameters[feature_name]
            )

        workspace.CreateNet(net)

        for feature_name in feature_value_map:
            workspace.FeedBlob(feature_name, feature_value_map[feature_name])
        workspace.RunNetOnce(net)

        for feature_name in feature_value_map:
            normalized_features = workspace.FetchBlob(
                feature_name + "_preprocessed"
            )
            tolerance = 0.01
            if feature_name == 'boxcox':
                # At the limit, boxcox has some numerical instability
                tolerance = 0.1
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
                ), '{} does not match: {} {}'.format(
                    feature_name, normalized_features[non_matching].tolist(),
                    test_features[feature_name][non_matching].tolist()
                )
            )

    def test_type_override(self):
        # Take a feature that should be identified as probability
        feature_value_map = preprocessing_util.read_data()
        probability_values = feature_value_map[identify_types.PROBABILITY]

        # And ask for a binary anyways
        parameter = normalization.identify_parameter(
            probability_values,
            feature_type=identify_types.BINARY
        )
        self.assertEqual(parameter.feature_type, 'BINARY')
