#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np
from typing import List, Dict, Optional, Tuple

from caffe2.python import workspace, core
import caffe2.proto.caffe2_pb2 as caffe2_pb2

from ml.rl.preprocessing import identify_types
from ml.rl.preprocessing.normalization import NormalizationParameters, \
    MISSING_VALUE
from ml.rl.preprocessing.identify_types import FEATURE_TYPES, ENUM

import logging
logger = logging.getLogger(__name__)


def sort_features_by_normalization(normalization_parameters):
    """
    Helper function to return a sorted list from a normalization map.
    Also returns the starting index for each feature type"""
    # Sort features by feature type
    sorted_features = []
    feature_starts = []
    for feature_type in FEATURE_TYPES:
        feature_starts.append(len(sorted_features))
        for feature in normalization_parameters.keys():
            norm = normalization_parameters[feature]
            if norm.feature_type == feature_type:
                sorted_features.append(feature)
    return sorted_features, feature_starts


class PreprocessorNet:
    def __init__(self, net: core.Net, clip_anomalies: bool) -> None:
        self.clip_anomalies = clip_anomalies

        self._net = net
        self.ONE = self._net.NextBlob('ONE')
        self.ZERO = self._net.NextBlob('ZERO')
        self.MISSING = self._net.NextBlob('MISSING_VALUE')
        self.MISSING_U = self._net.NextBlob('MISSING_VALUE_U')
        self.MISSING_L = self._net.NextBlob('MISSING_VALUE_L')
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
        self.MISSING_SCALAR = net.NextBlob('MISSING_SCALAR')
        workspace.FeedBlob(
            self.MISSING_SCALAR, np.array([MISSING_VALUE], dtype=np.float32)
        )
        net.GivenTensorFill(
            [], [self.MISSING_SCALAR], shape=[], values=[MISSING_VALUE]
        )
        self.parameters = [
            self.ZERO,
            self.ONE,
            self.MISSING,
            self.MISSING_L,
            self.MISSING_U,
            self.MISSING_SCALAR,
        ]

    def preprocess_blob(self, blob, normalization_parameters):
        """
        Takes in a blob and its normalization parameters. Outputs a tuple
        whose first element is a blob containing the normalized input blob
        and whose second element contains all the parameter blobs used to
        create it.

        Call this from a CPU context and ensure the input blob exists in it.
        """
        is_empty_u = self._net.NextBlob(blob + "__isempty_u")
        is_empty_l = self._net.NextBlob(blob + "__isempty_l")
        is_empty = self._net.NextBlob(blob + "__isempty")
        is_not_empty_bool = self._net.NextBlob(blob + "__isnotemptybool")
        is_not_empty = self._net.NextBlob(blob + "__isnotempty")
        output_blob = self._net.NextBlob(blob + "_preprocessed")
        zeros = self._net.NextBlob(blob + "_zeros")

        self._net.GT([blob, self.MISSING_L], [is_empty_l], broadcast=1)
        self._net.LT([blob, self.MISSING_U], [is_empty_u], broadcast=1)
        self._net.And([is_empty_l, is_empty_u], [is_empty])
        self._net.Not([is_empty], [is_not_empty_bool])
        self._net.Cast(
            [is_not_empty_bool], [is_not_empty],
            to=caffe2_pb2.TensorProto.FLOAT
        )
        for i in range(len(normalization_parameters) - 1):
            if normalization_parameters[
                i
            ].feature_type != normalization_parameters[i + 1].feature_type:
                raise Exception(
                    "Only one feature type is allowed per call to preprocess_blob!"
                )
        feature_type = normalization_parameters[0].feature_type
        parameters: List[str] = []
        if feature_type == identify_types.BINARY:
            is_gt_zero = self._net.NextBlob(blob + "__is_gt_zero")
            is_lt_zero = self._net.NextBlob(blob + "__is_lt_zero")
            self._net.GT([blob, self.ZERO], [is_gt_zero], broadcast=1)
            self._net.LT([blob, self.ZERO], [is_lt_zero], broadcast=1)
            bool_blob = self._net.NextBlob(blob + "__bool")
            self._net.Or([is_gt_zero, is_lt_zero], [bool_blob])
            self._net.Cast([bool_blob], [blob], to=caffe2_pb2.TensorProto.FLOAT)
        elif feature_type == identify_types.PROBABILITY:
            self._net.Clip([blob], [blob], min=0.01, max=0.99)
            self._net.Logit([blob], [blob])
        elif feature_type == identify_types.ENUM:
            if len(normalization_parameters) != 1:
                raise Exception(
                    "Only a single enum feature is allowed per call to preprocess_blob"
                )
            normalization_parameter = normalization_parameters[0]
            is_not_empty = self._net.NextBlob("{}__is_not_empty".format(blob))
            is_not_empty_cast = self._net.NextBlob(
                "{}__is_not_empty_cast".format(blob)
            )

            possible_values = normalization_parameter.possible_values
            for x in possible_values:
                if x < 0:
                    logger.fatal(
                        "Invalid enum possible value for feature " + blob + ": "
                        + str(x) + " " +
                        str(normalization_parameter.possible_values)
                    )
                    raise Exception(
                        "Invalid enum possible value for feature " + blob + ": "
                        + str(x) + " " +
                        str(normalization_parameter.possible_values)
                    )

            flat_blob = self._net.NextBlob('flat_blob')
            self._net.FlattenToVec([blob], [flat_blob])
            values_blob = self._net.NextBlob('values_blob')
            self._net.ConstantFill(
                [flat_blob],
                [values_blob],
                value=1.0,
                dtype=core.DataType.FLOAT,
            )
            one_length_blob = self._net.NextBlob('one_length_blob')
            self._net.ConstantFill(
                [flat_blob],
                [one_length_blob],
                value=1,
                dtype=core.DataType.INT32,
            )
            int_blob = self._net.NextBlob('int_blob')
            self._net.Cast(
                [flat_blob], [int_blob], to=caffe2_pb2.TensorProto.INT32
            )
            default_values = self._net.NextBlob('default_values')
            output_without_missing = self._net.NextBlob(
                'output_without_missing'
            )
            workspace.FeedBlob(default_values, np.array(0.0, dtype=np.float32))
            parameters.append(default_values)
            self._net.SparseToDenseMask(
                [int_blob, values_blob, default_values, one_length_blob],
                [output_without_missing],
                mask=list(possible_values),
                max_skipped_indices=int(1e9),
            )
            self._net.Not([is_empty], [is_not_empty])
            self._net.Cast(
                [is_not_empty], [is_not_empty_cast],
                to=caffe2_pb2.TensorProto.FLOAT
            )
            self._net.Mul(
                [output_without_missing, is_not_empty_cast], [output_blob],
                broadcast=1,
                axis=0
            )
            return output_blob, parameters
        elif feature_type == identify_types.QUANTILE:
            # This transformation replaces a set of values with their quantile.
            # The quantile boundaries are provided in the normalization params.

            quantile_blob = self._net.NextBlob('quantile_blob')
            num_boundaries_blob = self._net.NextBlob('num_boundaries_blob')
            quantile_sizes = [
                len(norm.quantiles) for norm in normalization_parameters
            ]
            workspace.FeedBlob(
                num_boundaries_blob, np.array(quantile_sizes, dtype=np.int32)
            )
            parameters.append(num_boundaries_blob)

            quantiles_blob = self._net.NextBlob('quantiles_blob')
            quantile_values = np.array([], dtype=np.float32)
            quantile_labels = np.array([], dtype=np.float32)
            for norm in normalization_parameters:
                quantile_values = np.append(
                    quantile_values, np.array(norm.quantiles, dtype=np.float32)
                )
                # TODO: Fix this: the np.unique is making this part not true.
                quantile_labels = np.append(
                    quantile_labels,
                    np.arange(len(norm.quantiles), dtype=np.float32) /
                    float(len(norm.quantiles))
                )
            quantiles = np.vstack([quantile_values, quantile_labels]).T
            workspace.FeedBlob(quantiles_blob, quantiles)
            parameters.append(quantiles_blob)

            self._net.Percentile(
                [blob, quantiles_blob, num_boundaries_blob], [quantile_blob]
            )
            blob = quantile_blob
        elif feature_type == identify_types.CONTINUOUS or \
                feature_type == identify_types.BOXCOX:
            boxcox_shifts = []
            boxcox_lambdas = []
            means = []
            stddevs = []

            for norm in normalization_parameters:
                if feature_type == identify_types.BOXCOX:
                    assert norm.boxcox_shift is not None and \
                        norm.boxcox_lambda is not None
                    boxcox_shifts.append(norm.boxcox_shift)
                    boxcox_lambdas.append(norm.boxcox_lambda)
                means.append(norm.mean)
                stddevs.append(norm.stddev)

            if feature_type == identify_types.BOXCOX:
                boxcox_shift = self._net.NextBlob(
                    '{}__boxcox_shift'.format(blob)
                )
                workspace.FeedBlob(
                    boxcox_shift, np.array(boxcox_shifts, dtype=np.float32)
                )
                parameters.append(boxcox_shift)
                boxcox_lambda = self._net.NextBlob(
                    '{}__boxcox_lambda'.format(blob)
                )
                workspace.FeedBlob(
                    boxcox_lambda, np.array(boxcox_lambdas, dtype=np.float32)
                )
                parameters.append(boxcox_lambda)

                self._net.BatchBoxCox(
                    [blob, boxcox_lambda, boxcox_shift], [blob]
                )

            means_blob = self._net.NextBlob('{}__preprocess_mean'.format(blob))
            workspace.FeedBlob(means_blob, np.array([means], dtype=np.float32))
            parameters.append(means_blob)
            stddevs_blob = self._net.NextBlob(
                '{}__preprocess_stddev'.format(blob)
            )
            workspace.FeedBlob(
                stddevs_blob, np.array([stddevs], dtype=np.float32)
            )
            parameters.append(stddevs_blob)
            self._net.Sub([blob, means_blob], [blob], broadcast=1, axis=0)
            self._net.Div([blob, stddevs_blob], [blob], broadcast=1, axis=0)
            if self.clip_anomalies:
                self._net.Clip([blob], [blob], min=-3.0, max=3.0)
        else:
            raise NotImplementedError(
                "Invalid feature type: {}".format(feature_type)
            )

        self._net.ConstantFill([blob], [zeros], value=0.)
        self._net.Mul([blob, is_not_empty], [output_blob])

        return output_blob, parameters

    def normalize_sparse_matrix(
        self,
        lengths_blob: str,
        keys_blob: str,
        values_blob: str,
        normalization_parameters: Dict[str, NormalizationParameters],
        blobname_prefix: str,
    ) -> Tuple[str, List[str]]:
        sorted_features, _ = sort_features_by_normalization(
            normalization_parameters
        )
        int_features = [int(feature) for feature in sorted_features]

        dense_input = self._net.NextBlob('dense_input')
        workspace.FeedBlob(dense_input, np.zeros(1, dtype=np.float32))
        self._net.SparseToDenseMask(
            [
                keys_blob,
                values_blob,
                self.MISSING_SCALAR,
                lengths_blob,
            ], [dense_input],
            mask=int_features
        )
        return self.normalize_dense_matrix(
            dense_input,
            sorted_features,
            normalization_parameters,
            blobname_prefix,
        )

    def normalize_dense_matrix(
        self,
        input_matrix: str,
        features: List[str],
        normalization_parameters: Dict[str, NormalizationParameters],
        blobname_prefix: str,
    ) -> Tuple[str, List[str]]:
        """
        Normalizes inputs according to parameters. Expects a dense matrix whose ith
        column corresponds to feature i.

        Note that the Caffe2 BatchBoxCox operator isn't implemented on CUDA GPU so
        we need to use a CPU context.

        :param input_matrix: Input matrix to normalize.
        :param features: Array that maps feature ids to column indices.
        :param normalization_parameters: Mapping from feature names to
            NormalizationParameters.
        :param blobname_prefix: Prefix for input blobs to norm_net.
        :param num_output_features: The number of features in an output processed
            datapoint. If set to None, this function will compute it.
        """
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            feature_starts = self._get_type_boundaries(
                features, normalization_parameters
            )

            normalized_input_blobs = []
            parameters: List[str] = []
            for i, feature_type in enumerate(FEATURE_TYPES):
                start_index = feature_starts[i]
                if (i + 1) == len(FEATURE_TYPES):
                    end_index = len(normalization_parameters)
                else:
                    end_index = feature_starts[i + 1]
                if start_index == end_index:
                    continue  # No features of this type
                if feature_type == ENUM:
                    # Each enum feature must be processed independently
                    for x in range(start_index, end_index):
                        sliced_input_features = self._get_input_blob(
                            blobname_prefix, feature_type, features[x]
                        )
                        self._net.Slice(
                            [input_matrix],
                            [sliced_input_features],
                            starts=[0, x],
                            ends=[-1, x + 1],
                        )
                        normalized_input_blob, blob_parameters = self.preprocess_blob(
                            sliced_input_features,
                            [normalization_parameters[features[x]]],
                        )
                        parameters.extend(blob_parameters)
                        normalized_input_blobs.append(normalized_input_blob)
                else:
                    sliced_input_features = self._get_input_blob(
                        blobname_prefix, feature_type, None
                    )
                    self._net.Slice(
                        [input_matrix],
                        [sliced_input_features],
                        starts=[0, start_index],
                        ends=[-1, end_index],
                    )
                    normalized_input_blob, blob_parameters = self.preprocess_blob(
                        sliced_input_features,
                        [
                            normalization_parameters[x]
                            for x in features[start_index:end_index]
                        ],
                    )
                    parameters.extend(blob_parameters)
                    normalized_input_blobs.append(normalized_input_blob)
            concatenated_input_blob = blobname_prefix + "_concatenated_input_blob"
            concatenated_input_blob_dim = blobname_prefix + \
                "_concatenated_input_blob_dim"
            for i, inp in enumerate(normalized_input_blobs):
                logger.info("input# {}: {}".format(i, inp))
            self._net.Concat(
                normalized_input_blobs,
                [concatenated_input_blob, concatenated_input_blob_dim],
                axis=1
            )
            self._net.NanCheck(concatenated_input_blob, concatenated_input_blob)
            return concatenated_input_blob, parameters

    def _get_type_boundaries(
        self,
        features: List[str],
        normalization_parameters: Dict[str, NormalizationParameters],
    ) -> List[int]:
        feature_starts = []
        on_feature_type = -1
        for i, feature in enumerate(features):
            feature_type = normalization_parameters[feature].feature_type
            feature_type_index = FEATURE_TYPES.index(feature_type)
            assert feature_type_index >= on_feature_type, \
                "Features are not sorted by feature type!"
            while feature_type_index > on_feature_type:
                feature_starts.append(i)
                on_feature_type += 1
        while on_feature_type < len(FEATURE_TYPES):
            feature_starts.append(len(features))
            on_feature_type += 1
        return feature_starts

    def _get_input_blob(
        self, prefix: str, feature_type: str, feature_id: Optional[str]
    ) -> str:
        if feature_type == ENUM:
            return "{}_{}_{}".format(prefix, feature_type, feature_id)
        else:
            return "{}_{}".format(prefix, feature_type)
