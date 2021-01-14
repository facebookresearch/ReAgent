#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import collections
import csv

import numpy as np
import six
from reagent.preprocessing import identify_types
from reagent.preprocessing.normalization import (
    BOX_COX_MARGIN,
    MAX_FEATURE_VALUE,
    MIN_FEATURE_VALUE,
    MISSING_VALUE,
    NormalizationParameters,
)
from scipy import special, stats


def default_normalizer(feats, min_value=None, max_value=None):
    normalization_types = [
        NormalizationParameters(
            feature_type="BINARY",
            boxcox_lambda=None,
            boxcox_shift=None,
            mean=0,
            stddev=1,
            possible_values=None,
            quantiles=None,
            min_value=min_value,
            max_value=max_value,
        ),
        NormalizationParameters(
            feature_type="PROBABILITY",
            boxcox_lambda=None,
            boxcox_shift=None,
            mean=0,
            stddev=1,
            possible_values=None,
            quantiles=None,
            min_value=min_value,
            max_value=max_value,
        ),
        NormalizationParameters(
            feature_type="CONTINUOUS",
            boxcox_lambda=None,
            boxcox_shift=None,
            mean=0,
            stddev=1,
            possible_values=None,
            quantiles=None,
            min_value=min_value,
            max_value=max_value,
        ),
        NormalizationParameters(
            feature_type="BOXCOX",
            boxcox_lambda=1,
            boxcox_shift=1,
            mean=0,
            stddev=1,
            possible_values=None,
            quantiles=None,
            min_value=min_value,
            max_value=max_value,
        ),
        NormalizationParameters(
            feature_type="QUANTILE",
            boxcox_lambda=None,
            boxcox_shift=None,
            mean=0,
            stddev=1,
            possible_values=None,
            quantiles=[0, 1],
            min_value=min_value,
            max_value=max_value,
        ),
        NormalizationParameters(
            feature_type="ENUM",
            boxcox_lambda=None,
            boxcox_shift=None,
            mean=0,
            stddev=1,
            possible_values=[0, 1],
            quantiles=None,
            min_value=min_value,
            max_value=max_value,
        ),
    ]
    normalization = collections.OrderedDict(
        [
            (feats[i], normalization_types[i % len(normalization_types)])
            for i in range(len(feats))
        ]
    )
    return normalization


def normalizer_helper(feats, feature_type, min_value=None, max_value=None):
    assert feature_type in (
        "DISCRETE_ACTION",
        "CONTINUOUS",
        "CONTINUOUS_ACTION",
    ), f"invalid feature type: {feature_type}."
    assert type(min_value) == type(max_value) and type(min_value) in (
        int,
        float,
        list,
        np.ndarray,
        type(None),
    ), f"invalid {type(min_value)}, {type(max_value)}"
    if type(min_value) in [int, float, type(None)]:
        min_value = [min_value] * len(feats)
        max_value = [max_value] * len(feats)
    normalization = collections.OrderedDict(
        [
            (
                feats[i],
                NormalizationParameters(
                    feature_type=feature_type,
                    boxcox_lambda=None,
                    boxcox_shift=None,
                    mean=0,
                    stddev=1,
                    possible_values=None,
                    quantiles=None,
                    min_value=float(min_value[i]) if min_value[i] is not None else None,
                    max_value=float(max_value[i]) if max_value[i] is not None else None,
                ),
            )
            for i in range(len(feats))
        ]
    )
    return normalization


def discrete_action_normalizer(feats):
    return normalizer_helper(feats, "DISCRETE_ACTION")


def only_continuous_normalizer(feats, min_value=None, max_value=None):
    return normalizer_helper(feats, "CONTINUOUS", min_value, max_value)


def only_continuous_action_normalizer(feats, min_value=None, max_value=None):
    return normalizer_helper(feats, "CONTINUOUS_ACTION", min_value, max_value)


def write_lists_to_csv(path, *args):
    rows = zip(*args)
    with open(path, "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


class NumpyFeatureProcessor(object):
    @staticmethod
    def value_to_quantile(original_value, quantiles):
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

    @classmethod
    def preprocess_feature(cls, feature, parameters):
        is_not_empty = 1 - np.isclose(feature, MISSING_VALUE)
        if parameters.feature_type == identify_types.BINARY:
            # Binary features are always 1 unless they are 0
            return ((feature != 0) * is_not_empty).astype(np.float32)
        if parameters.boxcox_lambda is not None:
            feature = stats.boxcox(
                np.maximum(feature + parameters.boxcox_shift, BOX_COX_MARGIN),
                parameters.boxcox_lambda,
            )
        # No *= to ensure consistent out-of-place operation.
        if parameters.feature_type == identify_types.PROBABILITY:
            feature = np.clip(feature, 0.01, 0.99)
            feature = special.logit(feature)
        elif parameters.feature_type == identify_types.QUANTILE:
            transformed_feature = np.zeros_like(feature)
            for i in six.moves.range(feature.shape[0]):
                transformed_feature[i] = cls.value_to_quantile(
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
                    # This check is required by the PT preprocessing but not C2
                    continue
                output_feature[i][mapping[val]] = 1.0
            return output_feature
        elif parameters.feature_type == identify_types.CONTINUOUS_ACTION:
            min_value = parameters.min_value
            max_value = parameters.max_value
            feature = (
                (feature - min_value) * ((1 - 1e-6) * 2 / (max_value - min_value))
                - 1
                + 1e-6
            )
        else:
            feature = feature - parameters.mean
            feature /= parameters.stddev
            feature = np.clip(feature, MIN_FEATURE_VALUE, MAX_FEATURE_VALUE)
        feature *= is_not_empty
        return feature

    @classmethod
    def preprocess(cls, features, parameters):
        result = {}
        for feature_name in features:
            result[feature_name] = cls.preprocess_feature(
                features[feature_name], parameters[feature_name]
            )
        return result

    @classmethod
    def preprocess_array(cls, arr, features, parameters):
        assert len(arr.shape) == 2 and arr.shape[1] == len(features)
        preprocessed_values = [
            cls.preprocess({f: v for f, v in zip(features, row)}, parameters)
            for row in arr
        ]
        return np.array(
            [[ex[f] for f in features] for ex in preprocessed_values], dtype=np.float32
        )
