#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np
from typing import List, Dict, Optional

from caffe2.python import workspace, core, dyndep
import caffe2.proto.caffe2_pb2 as caffe2_pb2

from ml.rl.preprocessing import identify_types
from ml.rl.preprocessing.normalization import NormalizationParameters,\
    get_num_output_features, MISSING_VALUE, BOX_COX_MIN_VALUE

import logging
logger = logging.getLogger(__name__)

dyndep.InitOpsLibrary('@/caffe2/caffe2/fb/operators:replace_values_op')


class PreprocessorNet:
    ONE = 'ONE'
    ZERO = 'ZERO'
    MISSING = 'MISSING_VALUE'
    MISSING_U = 'MISSING_VALUE_U'
    MISSING_L = 'MISSING_VALUE_L'

    def __init__(self, net: core.Net, clip_anomalies: bool) -> None:
        self.clip_anomalies = clip_anomalies

        self._net = net
        workspace.FeedBlob(self.ONE, np.array([1], dtype=np.float32))
        workspace.FeedBlob(self.ZERO, np.array([0], dtype=np.float32))
        workspace.FeedBlob(
            self.MISSING, np.array([MISSING_VALUE], dtype=np.float32)
        )
        workspace.FeedBlob(
            self.MISSING_U, np.array([MISSING_VALUE + 1e-4], dtype=np.float32)
        )
        workspace.FeedBlob(
            self.MISSING_L, np.array([MISSING_VALUE - 1e-4], dtype=np.float32)
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
        is_not_empty_bool = blob + "__isnotemptybool"
        is_not_empty = blob + "__isnotempty"
        output_blob = blob + "_preprocessed"

        self._net.GT([blob, self.MISSING_L], [is_empty_l], broadcast=1)
        self._net.LT([blob, self.MISSING_U], [is_empty_u], broadcast=1)
        self._net.And([is_empty_l, is_empty_u], [is_empty])
        self._net.Not([is_empty], [is_not_empty_bool])
        self._net.Cast(
            [is_not_empty_bool], [is_not_empty],
            to=caffe2_pb2.TensorProto.FLOAT
        )
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
        elif normalization_parameters.feature_type == identify_types.ENUM:
            possible_values_blobname = '{}__possible_values'.format(blob)
            blob_reshaped = "{}__reshaped".format(blob)
            blob_original_shape = "{}__original_shape".format(blob)
            output_blob_flattened = "{}__output_flattened".format(blob)
            output_blob_cast = "{}__output_cast".format(blob)
            index_size = "{}__index_size".format(blob)
            one_hot_out = "{}__one_hot_out".format(blob)
            is_not_empty = "{}__is_not_empty".format(blob)
            is_not_empty_cast = "{}__is_not_empty_cast".format(blob)

            possible_values = normalization_parameters.possible_values
            workspace.FeedBlob(
                possible_values_blobname,
                np.array(possible_values, dtype=np.float32)
            )
            workspace.FeedBlob(index_size, np.array(len(possible_values)))

            self._net.Reshape(
                [blob], [blob_reshaped, blob_original_shape], shape=[1, -1]
            )
            self._net.ReplaceValuesOp(
                [blob_reshaped, possible_values_blobname], [output_blob]
            )
            self._net.FlattenToVec([output_blob], [output_blob_flattened])
            self._net.Cast(
                [output_blob_flattened], [output_blob_cast],
                to=caffe2_pb2.TensorProto.INT64
            )
            self._net.OneHot([output_blob_cast, index_size], [one_hot_out])
            self._net.Cast(
                [one_hot_out], [blob], to=caffe2_pb2.TensorProto.FLOAT
            )
            self._net.Not([is_empty], [is_not_empty])
            self._net.Cast(
                [is_not_empty], [is_not_empty_cast],
                to=caffe2_pb2.TensorProto.FLOAT
            )
            self._net.Mul(
                [blob, is_not_empty_cast], [output_blob], broadcast=1, axis=0
            )
            return output_blob, parameters
        else:
            if normalization_parameters.boxcox_lambda is not None:
                boxcox_shift = '{}__boxcox_shift'.format(blob)
                workspace.FeedBlob(
                    boxcox_shift,
                    np.array(
                        normalization_parameters.boxcox_shift * -1,
                        dtype=np.float32
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
                self._net.Add([blob, boxcox_shift], [blob], broadcast=1, axis=0)
                self._net.Clip([blob], [blob], min=BOX_COX_MIN_VALUE)
                if abs(normalization_parameters.boxcox_lambda) < 1e-6:
                    self._net.Log([blob], [blob])
                else:
                    self._net.Pow(
                        [blob], [blob],
                        exponent=normalization_parameters.boxcox_lambda
                    )
                    self._net.Sub([blob, self.ONE], [blob], broadcast=1, axis=0)
                    self._net.Div(
                        [blob, boxcox_lambda], [blob], broadcast=1, axis=0
                    )
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

        if self.clip_anomalies:
            self._net.Clip([blob], [blob], min=-3.0, max=3.0)
        zeros = blob + "_zeros"
        self._net.ConstantFill([blob], [zeros], value=0.)
        self._net.Mul([blob, is_not_empty], [output_blob])

        return output_blob, parameters


def normalize_dense_matrix(
    inputs: np.ndarray,
    features: List[str],
    normalization_params: Dict[str, NormalizationParameters],
    norm_blob_map: Dict[int, str],
    norm_net: core.Net,
    blobname_template: str,
    num_output_features: Optional[int] = None,
) -> np.ndarray:
    """
    Normalizes inputs according to parameters. Expects a dense matrix whose ith
    column corresponds to feature i.

    Note that the Caffe2 BatchBoxCox operator isn't implemented on CUDA GPU so
    we need to use a CPU context.

    :param inputs: Numpy array with inputs to normalize. Should be of
        shape (any, num_features).
    :param features: Array of feature names.
    :param normalization_params: Mapping from feature names to
        NormalizationParameters.
    :param norm_blob_map: Dictionary that stores a mapping from feature index
        to input normalization blob name.
    :param norm_net: Caffe2 net for normalization.
    :param blobname_template: String template for input blobs to norm_net.
    :param num_output_features: The number of features in an output processed
        datapoint. If set to None, this function will compute it.
    """
    num_input_features = len(features)

    num_output_features = \
        num_output_features or get_num_output_features(normalization_params)

    assert inputs.shape[1] == num_input_features
    outputs = np.zeros((inputs.shape[0], num_output_features), dtype=np.float32)

    for idx in range(num_input_features):
        input_blob = blobname_template.format(idx)
        workspace.FeedBlob(input_blob, inputs[:, idx])
    workspace.RunNet(norm_net)

    output_col = 0
    for idx, feature in enumerate(features):
        normalized_input_blob = norm_blob_map[idx]
        normalized_inputs = workspace.FetchBlob(normalized_input_blob)
        normalization_param = normalization_params[feature]
        if normalization_param.feature_type == identify_types.ENUM:
            next_output_col = output_col + len(
                normalization_param.possible_values
            )
            outputs[:, output_col:next_output_col] = normalized_inputs
        else:
            next_output_col = output_col + 1
            outputs[:, output_col] = normalized_inputs
        output_col = next_output_col
    return outputs


def normalize_feature_map(
    feature_value_map: Dict[str, np.ndarray],
    norm_net: core.Net,
    features: List[str],
    norm_blob_map: Dict[int, str],
    blobname_template: str,
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
    features: List[str], blobname_template: str, clip_anomalies: bool
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
    preprocessor = PreprocessorNet(norm_net, clip_anomalies)
    for idx, feature in enumerate(features):
        input_blob = blobname_template.format(idx)
        reshaped_input_blob = input_blob + '_reshaped'
        original_shape = input_blob + '_original_shape'

        workspace.FeedBlob(input_blob, np.zeros(1, dtype=np.float32))
        norm_net.Reshape(
            [input_blob], [reshaped_input_blob, original_shape], shape=[-1, 1]
        )
        normalization_param = normalization_params[feature]
        normalized_input_blob, _ = preprocessor.preprocess_blob(
            reshaped_input_blob, normalization_param
        )
        norm_net.ReplaceNaN(normalized_input_blob, normalized_input_blob)
        if normalization_param.feature_type != identify_types.ENUM:
            norm_net.Reshape(
                [normalized_input_blob],
                [normalized_input_blob, original_shape],
                shape=[1, -1]
            )
        norm_blob_map[idx] = normalized_input_blob
    workspace.CreateNet(norm_net)
    return norm_blob_map
