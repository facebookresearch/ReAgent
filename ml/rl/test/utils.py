#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import collections
import csv
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import six
from ml.rl import types as rlt
from ml.rl.preprocessing import identify_types
from ml.rl.preprocessing.normalization import (
    BOX_COX_MARGIN,
    MAX_FEATURE_VALUE,
    MIN_FEATURE_VALUE,
    MISSING_VALUE,
    NormalizationParameters,
)
from scipy import special, stats


@dataclass
class ABIdFeatures(rlt.IdFeatureBase):
    a_id: rlt.ValueType
    b_id: rlt.ValueType

    @classmethod
    def get_feature_config(cls) -> Dict[str, rlt.IdFeatureConfig]:
        return {
            "a_id": rlt.IdFeatureConfig(feature_id=2002, id_mapping_name="a_mapping"),
            "b_id": rlt.IdFeatureConfig(feature_id=2003, id_mapping_name="b_mapping"),
        }


@dataclass
class CIdFeatures(rlt.IdFeatureBase):
    c_id: rlt.ValueType

    @classmethod
    def get_feature_config(cls) -> Dict[str, rlt.IdFeatureConfig]:
        return {
            "c_id": rlt.IdFeatureConfig(feature_id=2004, id_mapping_name="c_mapping")
        }


@dataclass
class IdOnlySequence(rlt.SequenceFeatureBase):
    id_features: ABIdFeatures

    @classmethod
    def get_max_length(cls) -> int:
        return 2


@dataclass
class IdAndFloatSequence(rlt.SequenceFeatureBase):
    id_features: CIdFeatures

    @classmethod
    def get_max_length(cls) -> int:
        return 3

    @classmethod
    def get_float_feature_infos(cls) -> List[rlt.FloatFeatureInfo]:
        return [
            rlt.FloatFeatureInfo(name="f{}".format(f_id), feature_id=f_id)
            for f_id in [1004]
        ]


@dataclass
class FloatOnlySequence(rlt.SequenceFeatureBase):
    @classmethod
    def get_max_length(cls) -> int:
        return 2

    @classmethod
    def get_float_feature_infos(cls) -> List[rlt.FloatFeatureInfo]:
        return [
            rlt.FloatFeatureInfo(name="f{}".format(f_id), feature_id=f_id)
            for f_id in [1001, 1002, 1003]
        ]


@dataclass
class SequenceFeatures(rlt.SequenceFeatures):
    id_only: IdOnlySequence
    id_and_float: IdAndFloatSequence
    float_only: FloatOnlySequence


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


def only_continuous_normalizer(feats, min_value=None, max_value=None):
    assert type(min_value) == type(max_value) and type(min_value) in (
        int,
        float,
        list,
        np.ndarray,
        type(None),
    )
    if type(min_value) in [int, float, type(None)]:
        min_value = [min_value] * len(feats)
        max_value = [max_value] * len(feats)
    normalization = collections.OrderedDict(
        [
            (
                feats[i],
                NormalizationParameters(
                    feature_type="CONTINUOUS",
                    boxcox_lambda=None,
                    boxcox_shift=None,
                    mean=0,
                    stddev=1,
                    possible_values=None,
                    quantiles=None,
                    min_value=min_value[i],
                    max_value=max_value[i],
                ),
            )
            for i in range(len(feats))
        ]
    )
    return normalization


def only_continuous_action_normalizer(feats, min_value=None, max_value=None):
    normalization = collections.OrderedDict(
        [
            (
                feats[i],
                NormalizationParameters(
                    feature_type="CONTINUOUS_ACTION",
                    boxcox_lambda=None,
                    boxcox_shift=None,
                    mean=0,
                    stddev=1,
                    possible_values=None,
                    quantiles=None,
                    min_value=min_value,
                    max_value=max_value,
                ),
            )
            for i in range(len(feats))
        ]
    )
    return normalization


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
