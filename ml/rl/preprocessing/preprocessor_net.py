#!/usr/bin/env python3


import numpy as np
from typing import List, Dict, Tuple

from caffe2.python import workspace, core
import caffe2.proto.caffe2_pb2 as caffe2_pb2

from ml.rl.caffe_utils import C2
from ml.rl.preprocessing import identify_types
from ml.rl.preprocessing.normalization import NormalizationParameters, \
    MISSING_VALUE
from ml.rl.preprocessing.identify_types import FEATURE_TYPES

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
            for parameter in normalization_parameters:
                possible_values = parameter.possible_values
                for x in possible_values:
                    if x < 0:
                        logger.fatal(
                            "Invalid enum possible value for feature: " +
                            str(x) + " " + str(parameter.possible_values)
                        )
                        raise Exception(
                            "Invalid enum possible value for feature " + blob +
                            ": " + str(x) + " " +
                            str(parameter.possible_values)
                        )

            int_blob = self._net.NextBlob('int_blob')
            self._net.Cast(
                [blob],
                [int_blob],
                to=core.DataType.INT32,
            )

            output_int_blob = self._net.NextBlob('output_int_blob')
            feature_lengths_blob = self._net.NextBlob('feature_lengths_blob')
            feature_values_blob = self._net.NextBlob('feature_values_blob')
            one_hot_output = self._net.NextBlob('one_hot_output')

            # Batch one hot transform with MISSING_VALUE as a possible value
            feature_lengths = [
                len(p.possible_values) + 1 for p in normalization_parameters
            ]
            workspace.FeedBlob(
                feature_lengths_blob,
                np.array(feature_lengths, dtype=np.int32),
            )

            feature_values = [
                x
                for p in normalization_parameters
                for x in p.possible_values + [int(MISSING_VALUE)]
            ]

            workspace.FeedBlob(
                feature_values_blob,
                np.array(feature_values, dtype=np.int32),
            )

            parameters.extend([feature_values_blob, feature_lengths_blob])

            self._net.BatchOneHot(
                [int_blob, feature_lengths_blob, feature_values_blob],
                [one_hot_output],
            )

            # Remove missing values with a mask
            flattened_one_hot = self._net.NextBlob('flattened_one_hot')
            self._net.FlattenToVec([one_hot_output], [flattened_one_hot])
            cols_to_include = [
                [1] * len(p.possible_values) + [0]
                for p in normalization_parameters
            ]
            cols_to_include = [x for col in cols_to_include for x in col]
            mask = self._net.NextBlob('mask')
            workspace.FeedBlob(mask, np.array(cols_to_include, dtype=np.int32))
            parameters.append(mask)

            zero_vec = self._net.NextBlob('zero_vec')
            self._net.ConstantFill(
                [one_hot_output], [zero_vec],
                value=0,
                dtype=caffe2_pb2.TensorProto.INT32
            )

            repeated_mask_int = self._net.NextBlob('repeated_mask_int')
            repeated_mask_bool = self._net.NextBlob('repeated_mask_bool')

            self._net.Add([zero_vec, mask], [repeated_mask_int], broadcast=1)
            self._net.Cast(
                [repeated_mask_int], [repeated_mask_bool],
                to=core.DataType.BOOL
            )

            flattened_repeated_mask = self._net.NextBlob(
                'flattened_repeated_mask'
            )
            self._net.FlattenToVec(
                [repeated_mask_bool], [flattened_repeated_mask]
            )

            flattened_one_hot_proc = self._net.NextBlob(
                'flattened_one_hot_proc'
            )
            self._net.BooleanMask(
                [flattened_one_hot, flattened_repeated_mask],
                [flattened_one_hot_proc, flattened_one_hot_proc + 'indices']
            )

            one_hot_shape = self._net.NextBlob('one_hot_shape')
            self._net.Shape([one_hot_output], [one_hot_shape])
            target_shape = self._net.NextBlob('target_shape')
            shape_delta = self._net.NextBlob('shape_delta')
            workspace.FeedBlob(
                shape_delta,
                np.array([0, len(normalization_parameters)], dtype=np.int64)
            )
            parameters.append(shape_delta)
            self._net.Sub(
                [one_hot_shape, shape_delta], [target_shape], broadcast=1
            )
            self._net.Reshape(
                [flattened_one_hot_proc, target_shape],
                [output_int_blob, output_int_blob + '_old_shape'],
            )

            self._net.Cast(
                [output_int_blob],
                [output_blob],
                to=core.DataType.FLOAT,
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

        dense_input, _ = C2.SparseToDenseMask(
            keys_blob,
            values_blob,
            self.MISSING_SCALAR,
            lengths_blob,
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
                sliced_input_features = self._get_input_blob(
                    blobname_prefix, feature_type
                )
                C2.net().Slice(
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
            for i, inp in enumerate(normalized_input_blobs):
                logger.info("input# {}: {}".format(i, inp))
            concatenated_input_blob, concatenated_input_blob_dim = C2.Concat(
                *normalized_input_blobs, axis=1
            )
            concatenated_input_blob = C2.NanCheck(concatenated_input_blob)
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

    def _get_input_blob(self, prefix: str, feature_type: str) -> str:
        return "{}_{}".format(prefix, feature_type)
