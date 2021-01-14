#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import json
import logging
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import reagent.types as rlt
import six
import torch
from reagent.parameters import NormalizationData, NormalizationParameters
from reagent.preprocessing import identify_types
from reagent.preprocessing.identify_types import DEFAULT_MAX_UNIQUE_ENUM, FEATURE_TYPES
from scipy import stats
from scipy.stats.mstats import mquantiles


logger = logging.getLogger(__name__)


BOX_COX_MAX_STDDEV = 1e8
BOX_COX_MARGIN = 1e-4
MISSING_VALUE = -1337.1337
DEFAULT_QUANTILE_K2_THRESHOLD = 1000.0
MINIMUM_SAMPLES_TO_IDENTIFY = 20
DEFAULT_MAX_QUANTILE_SIZE = 20
DEFAULT_NUM_SAMPLES = 100000
MAX_FEATURE_VALUE = 6.0
MIN_FEATURE_VALUE = MAX_FEATURE_VALUE * -1
EPS = 1e-6


def no_op_feature():
    return NormalizationParameters(
        identify_types.CONTINUOUS, None, 0, 0, 1, None, None, None, None
    )


def identify_parameter(
    feature_name,
    values,
    max_unique_enum_values=DEFAULT_MAX_UNIQUE_ENUM,
    quantile_size=DEFAULT_MAX_QUANTILE_SIZE,
    quantile_k2_threshold=DEFAULT_QUANTILE_K2_THRESHOLD,
    skip_box_cox=False,
    skip_quantiles=False,
    feature_type=None,
):
    if feature_type is None:
        feature_type = identify_types.identify_type(values, max_unique_enum_values)

    boxcox_lambda = None
    boxcox_shift = 0.0
    mean = 0.0
    stddev = 1.0
    possible_values = None
    quantiles = None
    assert feature_type in [
        identify_types.CONTINUOUS,
        identify_types.PROBABILITY,
        identify_types.BINARY,
        identify_types.ENUM,
        identify_types.CONTINUOUS_ACTION,
        identify_types.DISCRETE_ACTION,
        identify_types.DO_NOT_PREPROCESS,
    ], "unknown type {}".format(feature_type)
    assert (
        len(values) >= MINIMUM_SAMPLES_TO_IDENTIFY
    ), "insufficient information to identify parameter"

    min_value = float(np.min(values))
    max_value = float(np.max(values))

    if feature_type == identify_types.DO_NOT_PREPROCESS:
        mean = float(np.mean(values))
        values = values - mean
        stddev = max(float(np.std(values, ddof=1)), 1.0)
    if feature_type == identify_types.CONTINUOUS:
        if min_value == max_value:
            return no_op_feature()
        k2_original, p_original = stats.normaltest(values)

        # shift can be estimated but not in scipy
        boxcox_shift = float(min_value * -1)
        candidate_values, lambda_ = stats.boxcox(
            np.maximum(values + boxcox_shift, BOX_COX_MARGIN)
        )
        k2_boxcox, p_boxcox = stats.normaltest(candidate_values)
        logger.info(
            "Feature stats.  Original K2: {} P: {} Boxcox K2: {} P: {}".format(
                k2_original, p_original, k2_boxcox, p_boxcox
            )
        )
        if lambda_ < 0.9 or lambda_ > 1.1:
            # Lambda is far enough from 1.0 to be worth doing boxcox
            if k2_original > k2_boxcox * 10 and k2_boxcox <= quantile_k2_threshold:
                # The boxcox output is significantly more normally distributed
                # than the original data and is normal enough to apply
                # effectively.

                stddev = float(np.std(candidate_values, ddof=1))
                # Unclear whether this happens in practice or not
                if (
                    np.isfinite(stddev)
                    and stddev < BOX_COX_MAX_STDDEV
                    and not np.isclose(stddev, 0)
                ):
                    values = candidate_values
                    boxcox_lambda = float(lambda_)
        if boxcox_lambda is None or skip_box_cox:
            boxcox_shift = None
            boxcox_lambda = None
        if boxcox_lambda is not None:
            feature_type = identify_types.BOXCOX
        if (
            boxcox_lambda is None
            and k2_original > quantile_k2_threshold
            and (not skip_quantiles)
        ):
            feature_type = identify_types.QUANTILE
            quantiles = (
                np.unique(
                    mquantiles(
                        values,
                        np.arange(quantile_size + 1, dtype=np.float64)
                        / float(quantile_size),
                        alphap=0.0,
                        betap=1.0,
                    )
                )
                .astype(float)
                .tolist()
            )
            logger.info("Feature is non-normal, using quantiles: {}".format(quantiles))

    if (
        feature_type == identify_types.CONTINUOUS
        or feature_type == identify_types.BOXCOX
        or feature_type == identify_types.CONTINUOUS_ACTION
    ):
        mean = float(np.mean(values))
        values = values - mean
        stddev = max(float(np.std(values, ddof=1)), 1.0)
        if not np.isfinite(stddev):
            logger.info("Std. dev not finite for feature {}".format(feature_name))
            return None
        values /= stddev

    if feature_type == identify_types.ENUM:
        possible_values = np.unique(values.astype(int)).astype(int).tolist()

    return NormalizationParameters(
        feature_type,
        boxcox_lambda,
        boxcox_shift,
        mean,
        stddev,
        possible_values,
        quantiles,
        min_value,
        max_value,
    )


def get_feature_config(
    float_features: Optional[List[Tuple[int, str]]]
) -> rlt.ModelFeatureConfig:
    float_features = float_features or []
    float_feature_infos = [
        rlt.FloatFeatureInfo(name=f_name, feature_id=f_id)
        for f_id, f_name in float_features
    ]

    return rlt.ModelFeatureConfig(float_feature_infos=float_feature_infos)


def get_num_output_features(
    normalization_parameters: Dict[int, NormalizationParameters]
) -> int:
    return sum(
        map(
            lambda np: (
                len(np.possible_values) if np.feature_type == identify_types.ENUM else 1
            ),
            normalization_parameters.values(),
        )
    )


def get_feature_start_indices(
    sorted_features: List[int],
    normalization_parameters: Dict[int, NormalizationParameters],
):
    """ Returns the starting index for each feature in the output feature vector """
    start_indices = []
    cur_idx = 0
    for feature in sorted_features:
        np = normalization_parameters[feature]
        start_indices.append(cur_idx)
        if np.feature_type == identify_types.ENUM:
            assert np.possible_values is not None
            # pyre-fixme[6]: Expected `Sized` for 1st param but got
            #  `Optional[List[int]]`.
            cur_idx += len(np.possible_values)
        else:
            cur_idx += 1
    return start_indices


def sort_features_by_normalization(
    normalization_parameters: Dict[int, NormalizationParameters]
) -> Tuple[List[int], List[int]]:
    """
    Helper function to return a sorted list from a normalization map.
    Also returns the starting index for each feature type"""
    # Sort features by feature type
    sorted_features: List[int] = []
    feature_starts: List[int] = []
    assert isinstance(
        list(normalization_parameters.keys())[0], int
    ), "Normalization Parameters need to be int"
    for feature_type in FEATURE_TYPES:
        feature_starts.append(len(sorted_features))
        for feature in sorted(normalization_parameters.keys()):
            norm = normalization_parameters[feature]
            if norm.feature_type == feature_type:
                sorted_features.append(feature)
    return sorted_features, feature_starts


def deserialize(parameters_json) -> Dict[int, NormalizationParameters]:
    parameters = {}
    for feature, feature_parameters in six.iteritems(parameters_json):
        # Note: This is OK since NormalizationParameters is flat.
        params = NormalizationParameters(**json.loads(feature_parameters))
        # Check for negative enum IDs
        if params.feature_type == identify_types.ENUM:
            assert params.possible_values is not None
        parameters[int(feature)] = params
    return parameters


def serialize_one(feature_parameters):
    return json.dumps(asdict(feature_parameters))


def serialize(parameters):
    parameters_json = {}
    for feature, feature_parameters in six.iteritems(parameters):
        parameters_json[feature] = serialize_one(feature_parameters)
    return parameters_json


def get_feature_norm_metadata(feature_name, feature_value_list, norm_params):
    logger.info("Got feature: {}".format(feature_name))
    num_features = len(feature_value_list)
    if num_features < MINIMUM_SAMPLES_TO_IDENTIFY:
        return None

    feature_override = None
    if norm_params["feature_overrides"] is not None:
        feature_override = norm_params["feature_overrides"].get(feature_name, None)
    feature_override = feature_override or norm_params.get(
        "default_feature_override", None
    )

    feature_values = np.array(feature_value_list, dtype=np.float32)
    assert not (np.any(np.isinf(feature_values))), "Feature values contain infinity"
    assert not (
        np.any(np.isnan(feature_values))
    ), "Feature values contain nan (are there nulls in the feature values?)"
    normalization_parameters = identify_parameter(
        feature_name,
        feature_values,
        norm_params["max_unique_enum_values"],
        norm_params["quantile_size"],
        norm_params["quantile_k2_threshold"],
        norm_params["skip_box_cox"],
        norm_params["skip_quantiles"],
        feature_override,
    )
    logger.info(
        "Feature {} normalization: {}".format(feature_name, normalization_parameters)
    )
    return normalization_parameters


def construct_action_scale_tensor(action_norm_params, action_scale_overrides):
    """Construct tensors that will rescale each action value on each dimension i
    from [min_serving_value[i], max_serving_value[i]] to [-1, 1] for training.
    """
    sorted_features, _ = sort_features_by_normalization(action_norm_params)
    min_action_array = np.zeros((1, len(sorted_features)))
    max_action_array = np.zeros((1, len(sorted_features)))

    for idx, feature_id in enumerate(sorted_features):
        if feature_id in action_scale_overrides:
            min_action_array[0][idx] = action_scale_overrides[feature_id][0]
            max_action_array[0][idx] = action_scale_overrides[feature_id][1]
        else:
            min_action_array[0][idx] = action_norm_params[feature_id].min_value
            max_action_array[0][idx] = action_norm_params[feature_id].max_value

    min_action_range_tensor_serving = torch.from_numpy(min_action_array)
    max_action_range_tensor_serving = torch.from_numpy(max_action_array)
    return min_action_range_tensor_serving, max_action_range_tensor_serving
