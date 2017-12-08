#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np
from typing import List, Dict

from caffe2.python import workspace, core
import caffe2.proto.caffe2_pb2 as caffe2_pb2

from ml.rl.preprocessing import identify_types
from ml.rl.preprocessing.normalization import NormalizationParameters,\
    MISSING_VALUE, BOX_COX_MIN_VALUE

import logging
logger = logging.getLogger(__name__)


class PreprocessorNet:
    ZERO = 'ZERO'
    MISSING = 'MISSING_VALUE'
    MISSING_U = 'MISSING_VALUE_U'
    MISSING_L = 'MISSING_VALUE_L'

    def __init__(self, net: core.Net) -> None:
        # BatchBoxCox isn't implemented on CUDA GPU so we need to use CPU
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            self._net = net
            workspace.FeedBlob(self.ZERO, np.array([0], dtype=np.float32))
            workspace.FeedBlob(
                self.MISSING, np.array([MISSING_VALUE], dtype=np.float32)
            )
            workspace.FeedBlob(
                self.MISSING_U,
                np.array([MISSING_VALUE + 1e-4], dtype=np.float32)
            )
            workspace.FeedBlob(
                self.MISSING_L,
                np.array([MISSING_VALUE - 1e-4], dtype=np.float32)
            )
            self.parameters = [
                self.ZERO, self.MISSING, self.MISSING_L, self.MISSING_U
            ]

    def preprocess_blob(self, blob, normalization_parameters):
        """
        Takes in a blob and its normalization parameters. Outputs a tuple
        whose first element is a blob containing the normalized input blob
        and whose second element contains all the parameter blobs used to
        create it.

        Call this from a CPU context and ensure the input blob exists in it.
        """
        is_empty_u = blob + "__isempty_u"
        is_empty_l = blob + "__isempty_l"
        is_empty = blob + "__isempty"
        self._net.GT([blob, self.MISSING_L], [is_empty_l], broadcast=1)
        self._net.LT([blob, self.MISSING_U], [is_empty_u], broadcast=1)
        self._net.And([is_empty_l, is_empty_u], [is_empty])
        parameters: List[str] = []
        if normalization_parameters.feature_type == identify_types.BINARY:
            is_gt_zero = blob + "__is_gt_zero"
            is_lt_zero = blob + "__is_lt_zero"
            self._net.GT([blob, self.ZERO], [is_gt_zero], broadcast=1)
            self._net.LT([blob, self.ZERO], [is_lt_zero], broadcast=1)
            bool_blob = blob + "__bool"
            self._net.Or([is_gt_zero, is_lt_zero], [bool_blob])
            self._net.Cast([bool_blob], [blob], to=caffe2_pb2.TensorProto.FLOAT)
        elif normalization_parameters.feature_type == identify_types.PROBABILITY:
            self._net.Clip([blob], [blob], min=0.01, max=0.99)
            self._net.Logit([blob], [blob])
        else:
            if normalization_parameters.boxcox_lambda is not None:
                boxcox_shift = '{}__boxcox_shift'.format(blob)
                workspace.FeedBlob(
                    boxcox_shift,
                    np.array(
                        normalization_parameters.boxcox_shift, dtype=np.float32
                    )
                )
                boxcox_lambda = '{}__boxcox_lambda'.format(blob)
                workspace.FeedBlob(
                    boxcox_lambda,
                    np.array(
                        normalization_parameters.boxcox_lambda,
                        dtype=np.float32
                    )
                )
                self._net.Sub([blob, boxcox_shift], [blob], broadcast=1, axis=0)
                self._net.Clip([blob], [blob], min=BOX_COX_MIN_VALUE)
                self._net.BatchBoxCox([blob, boxcox_lambda, self.ZERO], [blob])
                parameters = [boxcox_lambda, boxcox_shift]

            mean = '{}__preprocess_mean'.format(blob)
            workspace.FeedBlob(
                mean, np.array(normalization_parameters.mean, dtype=np.float32)
            )
            stddev = '{}__preprocess_stddev'.format(blob)
            workspace.FeedBlob(
                stddev,
                np.array(normalization_parameters.stddev, dtype=np.float32)
            )
            self._net.Sub([blob, mean], [blob], broadcast=1, axis=0)
            self._net.Div([blob, stddev], [blob], broadcast=1, axis=0)
            parameters = parameters + [mean, stddev]

        zeros = blob + "_zeros"
        self._net.ConstantFill([blob], [zeros], value=0.)
        output_blob = blob + "_preprocessed"
        self._net.Where([is_empty, zeros, blob], [output_blob])
        self._net.ConstantFill([blob], [blob], value=MISSING_VALUE)

        return output_blob, parameters


def normalize_dense_matrix(
    inputs: np.ndarray, num_features: int, norm_blob_map: Dict[int, str],
    norm_net: core.Net, blobname_template: str
) -> np.ndarray:
    """
    Normalizes inputs according to parameters. Expects a dense matrix whose ith
    column corresponds to feature i.

    Note that the Caffe2 BatchBoxCox operator isn't implemented on CUDA GPU so
    we need to use a CPU context.

    :param inputs: Numpy array with inputs to normalize. Should be of
        shape (any, num_features).
    :param num_features: Integer number of features.
    :param norm_blob_map: Dictionary that stores a mapping from feature index
        to input normalization blob name.
    :param norm_net: Caffe2 net for normalization.
    :param blobname_template: String template for input blobs to norm_net.
    """
    assert inputs.shape[1] == num_features
    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
        for idx in range(num_features):
            input_blob = blobname_template.format(idx)
            workspace.FeedBlob(input_blob, inputs[:, idx])
        workspace.RunNet(norm_net)
        for idx in range(num_features):
            normalized_input_blob = norm_blob_map[idx]
            normalized_inputs = workspace.FetchBlob(normalized_input_blob)
            inputs[:, idx] = normalized_inputs
    return inputs


def normalize_feature_map(
    feature_value_map: Dict[str, np.ndarray], norm_net: core.Net,
    features: List[str], norm_blob_map: Dict[int, str], blobname_template: str
):
    """
    Normalizes the features in feature_value_map and returns another dictionary
    whose values are the normalizes features in feature_value_map.

    :param feature_value_map: Stores inputs to normalize. Maps from feature name
        to feature value.
    :param norm_net: Caffe2 net for normalization.
    :param features: Array of feature names.
    :param norm_blob_map: Dictionary that stores a mapping from feature index
        to input normalization blob name.
    :param blobname_template: String template for input blobs to norm_net.
    """
    normalized_features = {}
    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
        for idx, feature in enumerate(features):
            input_blob = blobname_template.format(idx)
            workspace.FeedBlob(
                input_blob, feature_value_map[feature].astype(np.float32)
            )
        workspace.RunNet(norm_net)
        for idx, feature_name in enumerate(features):
            normalized_input_blob = norm_blob_map[idx]
            normalized_features[feature_name
                               ] = workspace.FetchBlob(normalized_input_blob)
    return normalized_features


def prepare_normalization(
    norm_net: core.Net,
    normalization_params: Dict[str, NormalizationParameters],
    features: List[str], blobname_template: str
) -> Dict[int, str]:
    """
    Sets up operators for normalization net and returns a mapping from feature
    index to input blob name for the net.

    Note that the Caffe2 BatchBoxCox operator isn't implemented on CUDA GPU so
    we need to use a CPU context.

    :param norm_net: Caffe2 net for normalization.
    :param normalization_params: Mapping from feature names to
        NormalizationParameters.
    :param features: Array of feature names.
    :param blobname_template: String template for input blobs to norm_net.
    """
    norm_blob_map = {}
    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
        preprocessor = PreprocessorNet(norm_net)
        for idx, feature in enumerate(features):
            input_blob = blobname_template.format(idx)
            reshaped_input_blob = input_blob + '_reshaped'
            original_shape = input_blob + '_original_shape'

            workspace.FeedBlob(input_blob, np.zeros(1, dtype=np.float32))
            norm_net.Reshape(
                [input_blob], [reshaped_input_blob, original_shape],
                shape=[-1, 1]
            )
            normalized_input_blob, _ = preprocessor.preprocess_blob(
                reshaped_input_blob, normalization_params[feature]
            )
            norm_net.ReplaceNaN(normalized_input_blob, normalized_input_blob)
            norm_net.Reshape(
                [normalized_input_blob],
                [normalized_input_blob, original_shape],
                shape=[1, -1]
            )
            norm_blob_map[idx] = normalized_input_blob
        workspace.CreateNet(norm_net)
    return norm_blob_map
