#!/usr/bin/env python3


import logging
from typing import Dict, List, Tuple

import caffe2.proto.caffe2_pb2 as caffe2_pb2
import numpy as np
from caffe2.python import core, workspace
from ml.rl.caffe_utils import C2
from ml.rl.preprocessing import identify_types
from ml.rl.preprocessing.identify_types import FEATURE_TYPES
from ml.rl.preprocessing.normalization import MISSING_VALUE, NormalizationParameters


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
        for feature in sorted(normalization_parameters.keys()):
            norm = normalization_parameters[feature]
            if norm.feature_type == feature_type:
                sorted_features.append(feature)
    return sorted_features, feature_starts


class PreprocessorNet:
    def __init__(self, clip_anomalies: bool) -> None:
        self.clip_anomalies = clip_anomalies

    def preprocess_blob(self, blob, normalization_parameters):
        """
        Takes in a blob and its normalization parameters. Outputs a tuple
        whose first element is a blob containing the normalized input blob
        and whose second element contains all the parameter blobs used to
        create it.

        Call this from a CPU context and ensure the input blob exists in it.
        """

        parameters: List[str] = []

        ZERO = self._store_parameter(
            parameters, "ZERO", np.array([0], dtype=np.float32)
        )

        MISSING_U = self._store_parameter(
            parameters, "MISSING_U", np.array([MISSING_VALUE + 1e-4], dtype=np.float32)
        )
        MISSING_L = self._store_parameter(
            parameters, "MISSING_L", np.array([MISSING_VALUE - 1e-4], dtype=np.float32)
        )

        is_empty_l = C2.GT(blob, MISSING_L, broadcast=1)
        is_empty_u = C2.LT(blob, MISSING_U, broadcast=1)
        is_empty = C2.And(is_empty_l, is_empty_u)

        for i in range(len(normalization_parameters) - 1):
            if (
                normalization_parameters[i].feature_type
                != normalization_parameters[i + 1].feature_type
            ):
                raise Exception(
                    "Only one feature type is allowed per call to preprocess_blob!"
                )
        feature_type = normalization_parameters[0].feature_type
        if feature_type == identify_types.BINARY:
            TOLERANCE = self._store_parameter(
                parameters, "TOLERANCE", np.array(1e-3, dtype=np.float32)
            )
            is_gt_zero = C2.GT(blob, C2.Add(ZERO, TOLERANCE, broadcast=1), broadcast=1)
            is_lt_zero = C2.LT(blob, C2.Sub(ZERO, TOLERANCE, broadcast=1), broadcast=1)
            bool_blob = C2.Or(is_gt_zero, is_lt_zero)
            blob = C2.Cast(bool_blob, to=caffe2_pb2.TensorProto.FLOAT)
        elif feature_type == identify_types.PROBABILITY:
            blob = C2.Logit(C2.Clip(blob, min=0.01, max=0.99))
        elif feature_type == identify_types.ENUM:
            for parameter in normalization_parameters:
                possible_values = parameter.possible_values
                for x in possible_values:
                    if x < 0:
                        logger.fatal(
                            "Invalid enum possible value for feature: "
                            + str(x)
                            + " "
                            + str(parameter.possible_values)
                        )
                        raise Exception(
                            "Invalid enum possible value for feature "
                            + blob
                            + ": "
                            + str(x)
                            + " "
                            + str(parameter.possible_values)
                        )

            int_blob = C2.Cast(blob, to=core.DataType.INT32)

            # Batch one hot transform with MISSING_VALUE as a possible value
            feature_lengths = [
                len(p.possible_values) + 1 for p in normalization_parameters
            ]
            feature_lengths_blob = self._store_parameter(
                parameters,
                "feature_lengths_blob",
                np.array(feature_lengths, dtype=np.int32),
            )

            feature_values = [
                x
                for p in normalization_parameters
                for x in p.possible_values + [int(MISSING_VALUE)]
            ]
            feature_values_blob = self._store_parameter(
                parameters,
                "feature_values_blob",
                np.array(feature_values, dtype=np.int32),
            )

            one_hot_output = C2.BatchOneHot(
                int_blob, feature_lengths_blob, feature_values_blob
            )
            flattened_one_hot = C2.FlattenToVec(one_hot_output)

            # Remove missing values with a mask
            cols_to_include = [
                [1] * len(p.possible_values) + [0] for p in normalization_parameters
            ]
            cols_to_include = [x for col in cols_to_include for x in col]
            mask = self._store_parameter(
                parameters, "mask", np.array(cols_to_include, dtype=np.int32)
            )

            zero_vec = C2.ConstantFill(
                one_hot_output, value=0, dtype=caffe2_pb2.TensorProto.INT32
            )

            repeated_mask_bool = C2.Cast(
                C2.Add(zero_vec, mask, broadcast=1), to=core.DataType.BOOL
            )

            flattened_repeated_mask = C2.FlattenToVec(repeated_mask_bool)

            flattened_one_hot_proc = C2.NextBlob("flattened_one_hot_proc")
            flattened_one_hot_proc_indices = C2.NextBlob(
                "flattened_one_hot_proc_indices"
            )
            C2.net().BooleanMask(
                [flattened_one_hot, flattened_repeated_mask],
                [flattened_one_hot_proc, flattened_one_hot_proc_indices],
            )

            one_hot_shape = C2.Shape(one_hot_output)

            shape_delta = self._store_parameter(
                parameters,
                "shape_delta",
                np.array([0, len(normalization_parameters)], dtype=np.int64),
            )

            target_shape = C2.Sub(one_hot_shape, shape_delta, broadcast=1)
            output_int_blob = C2.NextBlob("output_int_blob")
            output_int_blob_old_shape = C2.NextBlob("output_int_blob_old_shape")
            C2.net().Reshape(
                [flattened_one_hot_proc, target_shape],
                [output_int_blob, output_int_blob_old_shape],
            )

            output_blob = C2.Cast(output_int_blob, to=core.DataType.FLOAT)

            return output_blob, parameters
        elif feature_type == identify_types.QUANTILE:
            # This transformation replaces a set of values with their quantile.
            # The quantile boundaries are provided in the normalization params.

            quantile_sizes = [len(norm.quantiles) for norm in normalization_parameters]
            num_boundaries_blob = self._store_parameter(
                parameters,
                "num_boundaries_blob",
                np.array(quantile_sizes, dtype=np.int32),
            )

            quantile_values = np.array([], dtype=np.float32)
            quantile_labels = np.array([], dtype=np.float32)
            for norm in normalization_parameters:
                quantile_values = np.append(
                    quantile_values, np.array(norm.quantiles, dtype=np.float32)
                )
                # TODO: Fix this: the np.unique is making this part not true.
                quantile_labels = np.append(
                    quantile_labels,
                    np.arange(len(norm.quantiles), dtype=np.float32)
                    / float(len(norm.quantiles)),
                )
            quantiles = np.vstack([quantile_values, quantile_labels]).T
            quantiles_blob = self._store_parameter(
                parameters, "quantiles_blob", quantiles
            )

            quantile_blob = C2.Percentile(blob, quantiles_blob, num_boundaries_blob)
            blob = quantile_blob
        elif (
            feature_type == identify_types.CONTINUOUS
            or feature_type == identify_types.BOXCOX
        ):
            boxcox_shifts = []
            boxcox_lambdas = []
            means = []
            stddevs = []

            for norm in normalization_parameters:
                if feature_type == identify_types.BOXCOX:
                    assert (
                        norm.boxcox_shift is not None and norm.boxcox_lambda is not None
                    )
                    boxcox_shifts.append(norm.boxcox_shift)
                    boxcox_lambdas.append(norm.boxcox_lambda)
                means.append(norm.mean)
                stddevs.append(norm.stddev)

            if feature_type == identify_types.BOXCOX:
                boxcox_shift_blob = self._store_parameter(
                    parameters,
                    "boxcox_shift",
                    np.array(boxcox_shifts, dtype=np.float32),
                )
                boxcox_lambda_blob = self._store_parameter(
                    parameters,
                    "boxcox_shift",
                    np.array(boxcox_lambdas, dtype=np.float32),
                )

                blob = C2.BatchBoxCox(blob, boxcox_lambda_blob, boxcox_shift_blob)

            means_blob = self._store_parameter(
                parameters, "means_blob", np.array([means], dtype=np.float32)
            )
            stddevs_blob = self._store_parameter(
                parameters, "stddevs_blob", np.array([stddevs], dtype=np.float32)
            )

            blob = C2.Sub(blob, means_blob, broadcast=1, axis=0)
            blob = C2.Div(blob, stddevs_blob, broadcast=1, axis=0)
            if self.clip_anomalies:
                blob = C2.Clip(blob, min=-3.0, max=3.0)
        else:
            raise NotImplementedError("Invalid feature type: {}".format(feature_type))

        zeros = C2.ConstantFill(blob, value=0.)
        output_blob = C2.Where(is_empty, zeros, blob)

        return output_blob, parameters

    def _store_parameter(self, parameters, name, value):
        c2_name = C2.NextBlob(name)
        workspace.FeedBlob(c2_name, value)
        parameters.append(c2_name)
        return c2_name

    def normalize_sparse_matrix(
        self,
        lengths_blob: str,
        keys_blob: str,
        values_blob: str,
        normalization_parameters: Dict[str, NormalizationParameters],
        blobname_prefix: str,
        split_sparse_to_dense: bool,
        split_expensive_feature_groups: bool,
        normalize: bool = True,
    ) -> Tuple[str, List[str]]:
        sorted_features, _ = sort_features_by_normalization(normalization_parameters)
        int_features = [int(feature) for feature in sorted_features]

        preprocess_num_batches = 8 if split_sparse_to_dense else 1

        lengths_batch = []
        keys_batch = []
        values_batch = []
        for _ in range(preprocess_num_batches):
            lengths_batch.append(C2.NextBlob(blobname_prefix + "_length_batch"))
            keys_batch.append(C2.NextBlob(blobname_prefix + "_key_batch"))
            values_batch.append(C2.NextBlob(blobname_prefix + "_value_batch"))

        C2.net().Split([lengths_blob], lengths_batch, axis=0)
        total_lengths_batch = []
        for x in range(preprocess_num_batches):
            total_lengths_batch.append(
                C2.Reshape(
                    C2.ReduceBackSum(lengths_batch[x], num_reduce_dims=1), shape=[1]
                )[0]
            )
        total_lengths_batch_concat, _ = C2.Concat(*total_lengths_batch, axis=0)
        C2.net().Split([keys_blob, total_lengths_batch_concat], keys_batch, axis=0)
        C2.net().Split([values_blob, total_lengths_batch_concat], values_batch, axis=0)

        dense_input_fragments = []
        parameters: List[str] = []

        MISSING_SCALAR = self._store_parameter(
            parameters, "MISSING_SCALAR", np.array([MISSING_VALUE], dtype=np.float32)
        )
        C2.net().GivenTensorFill([], [MISSING_SCALAR], shape=[], values=[MISSING_VALUE])

        for preprocess_batch in range(preprocess_num_batches):
            dense_input_fragment = C2.SparseToDenseMask(
                keys_batch[preprocess_batch],
                values_batch[preprocess_batch],
                MISSING_SCALAR,
                lengths_batch[preprocess_batch],
                mask=int_features,
            )[0]

            if normalize:
                normalized_fragment, p = self.normalize_dense_matrix(
                    dense_input_fragment,
                    sorted_features,
                    normalization_parameters,
                    blobname_prefix,
                    split_expensive_feature_groups,
                )
                dense_input_fragments.append(normalized_fragment)
                parameters.extend(p)
            else:
                dense_input_fragments.append(dense_input_fragment)

        dense_input = C2.NextBlob(blobname_prefix + "_dense_input")
        dense_input_dims = C2.NextBlob(blobname_prefix + "_dense_input_dims")
        C2.net().Concat(dense_input_fragments, [dense_input, dense_input_dims], axis=0)

        return dense_input, parameters

    def normalize_dense_matrix(
        self,
        input_matrix: str,
        features: List[str],
        normalization_parameters: Dict[str, NormalizationParameters],
        blobname_prefix: str,
        split_expensive_feature_groups: bool = False,
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
                slices = []

                split_feature_group, split_intervals = self._should_split_feature_group(
                    split_expensive_feature_groups, start_index, end_index, feature_type
                )

                if split_feature_group:
                    for j in range(len(split_intervals) - 1):
                        slice_blob = self._get_input_blob_indexed(
                            blobname_prefix, feature_type, j
                        )
                        C2.net().Slice(
                            [input_matrix],
                            [slice_blob],
                            starts=[0, split_intervals[j]],
                            ends=[-1, split_intervals[j + 1]],
                        )
                        slices.append(
                            (slice_blob, split_intervals[j], split_intervals[j + 1])
                        )
                else:
                    sliced_input_features = self._get_input_blob(
                        blobname_prefix, feature_type
                    )

                    C2.net().Slice(
                        [input_matrix],
                        [sliced_input_features],
                        starts=[0, start_index],
                        ends=[-1, end_index],
                    )

                    slices.append((sliced_input_features, start_index, end_index))

                for (slice_blob, start, end) in slices:
                    normalized_input_blob, blob_parameters = self.preprocess_blob(
                        slice_blob,
                        [normalization_parameters[x] for x in features[start:end]],
                    )
                    logger.info(
                        "Processed split ({}, {}) for feature type {}".format(
                            start, end, feature_type
                        )
                    )
                    parameters.extend(blob_parameters)
                    normalized_input_blobs.append(normalized_input_blob)
            for i, inp in enumerate(normalized_input_blobs):
                logger.info("input# {}: {}".format(i, inp))
            concatenated_input_blob, concatenated_input_blob_dim = C2.Concat(
                *normalized_input_blobs, axis=1
            )
            return concatenated_input_blob, parameters

    def concat_states_and_possible_actions(
        self,
        state_preprocessed_matrix_blob: str,
        possible_actions_blob: str,
        possible_actions_lengths_blob: str,
    ) -> str:
        stacked_states = C2.LengthsTile(
            state_preprocessed_matrix_blob, possible_actions_lengths_blob
        )
        state_action_pairs, _ = C2.Concat(stacked_states, possible_actions_blob, axis=1)
        return state_action_pairs

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
            assert (
                feature_type_index >= on_feature_type
            ), "Features are not sorted by feature type!"
            while feature_type_index > on_feature_type:
                feature_starts.append(i)
                on_feature_type += 1
        while on_feature_type < len(FEATURE_TYPES):
            feature_starts.append(len(features))
            on_feature_type += 1
        return feature_starts

    def _get_input_blob(self, prefix: str, feature_type: str) -> str:
        return "{}_{}".format(prefix, feature_type)

    def _get_input_blob_indexed(self, prefix: str, feature_type: str, idx: int) -> str:
        return "{}_{}_{}".format(prefix, feature_type, idx)

    def _should_split_feature_group(
        self,
        split_expensive_feature_groups: bool,
        start_index: int,
        end_index: int,
        feature_type: str,
    ) -> Tuple[bool, List[int]]:
        """
        Since this net is CPU bound, split into independent groups, so that
        the preprocessing can be parallelized while training.
        """
        if not split_expensive_feature_groups:
            return False, []
        if feature_type in [identify_types.ENUM, identify_types.QUANTILE]:
            if (end_index - start_index) > 32:
                step = (end_index - start_index) // 7
                intervals = list(range(start_index, end_index, step)) + [end_index]
                return True, intervals
        return False, []
