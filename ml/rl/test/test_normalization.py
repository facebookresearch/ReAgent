from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io

import numpy as np
from scipy import special
import six
import unittest

from caffe2.python import core, workspace
from ml.rl.preprocessing import normalization
from ml.rl.preprocessing import identify_types
from ml.rl.preprocessing.preprocessor_net import \
    PreprocessorNet, MISSING_VALUE
from ml.rl.test import preprocessing_util


class TestNormalization(unittest.TestCase):
    def test_normalization(self):
        feature_value_map = preprocessing_util.read_data()

        types = identify_types.identify_types(feature_value_map)
        types_dict = identify_types.identify_types_dict(feature_value_map)
        normalization_parameters = normalization.identify_parameters(
            feature_value_map, types_dict
        )
        normalized_features = normalization.preprocess(
            feature_value_map, normalization_parameters
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
            if normalization_parameters[
                k
            ].feature_type == identify_types.PROBABILITY:
                sigmoidv = special.expit(v)
                self.assertTrue(
                    np.all(
                        np.logical_and(
                            np.greater(sigmoidv, 0), np.less(sigmoidv, 1)
                        )
                    )
                )
            else:
                one_stddev = np.isclose(np.std(v, axis=0, ddof=1), 1)
                zero_stddev = np.isclose(np.std(v, axis=0, ddof=1), 0)
                zero_mean = np.isclose(np.mean(v, axis=0), 0)
                is_binary = types[k] == identify_types.BINARY
                self.assertTrue(np.all(np.logical_or(zero_mean, is_binary)))
                self.assertTrue(
                    np.all(
                        np.logical_or(
                            np.logical_or(one_stddev, zero_stddev), is_binary
                        )
                    )
                )

                has_boxcox = normalization_parameters[k
                                                     ].boxcox_lambda is not None
                is_ctd = types[k] == identify_types.CONTINUOUS
                # This should be true at the moment
                self.assertTrue(is_ctd == has_boxcox)

    def test_persistency(self):
        feature_value_map = preprocessing_util.read_data()

        types = identify_types.identify_types_dict(feature_value_map)
        normalization_parameters = normalization.identify_parameters(
            feature_value_map, types
        )

        with io.StringIO() as f:
            normalization.write_parameters(f, normalization_parameters)
            f.seek(0)
            read_parameters = normalization.load_parameters(f)

        self.assertEqual(read_parameters, normalization_parameters)

    def test_preprocessing_network(self):
        feature_value_map = preprocessing_util.read_data()
        types = identify_types.identify_types_dict(feature_value_map)
        normalization_parameters = normalization.identify_parameters(
            feature_value_map, types
        )
        test_features = normalization.preprocess(
            feature_value_map, normalization_parameters
        )
        test_features[u'186'] = 0

        net = core.Net("PreprocessingTestNet")
        preprocessor = PreprocessorNet(net)
        for feature_name in feature_value_map:
            workspace.FeedBlob(feature_name, np.array([0], dtype=np.int32))
            preprocessor.preprocess_blob(
                feature_name, normalization_parameters[feature_name]
            )

        workspace.CreateNet(net)

        for feature_name in feature_value_map:
            if feature_name != u'186':
                workspace.FeedBlob(
                    feature_name,
                    feature_value_map[feature_name].astype(np.float32)
                )
            else:
                workspace.FeedBlob(
                    feature_name, MISSING_VALUE * np.ones(1, dtype=np.float32)
                )
        workspace.RunNetOnce(net)

        for feature_name in feature_value_map:
            normalized_features = workspace.FetchBlob(
                feature_name + "_preprocessed"
            )
            self.assertTrue(
                np.all(
                    np.
                    isclose(normalized_features, test_features[feature_name])
                )
            )
        for feature_name in feature_value_map:
            if feature_name != u'186':
                workspace.FeedBlob(
                    feature_name,
                    feature_value_map[feature_name].astype(np.float32)
                )
        workspace.RunNetOnce(net)

        for feature_name in feature_value_map:
            normalized_features = workspace.FetchBlob(
                feature_name + "_preprocessed"
            )

            self.assertTrue(
                np.all(
                    np.
                    isclose(normalized_features, test_features[feature_name])
                )
            )
