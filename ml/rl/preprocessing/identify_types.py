#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np


BINARY = "BINARY"
PROBABILITY = "PROBABILITY"
CONTINUOUS = "CONTINUOUS"
BOXCOX = "BOXCOX"
ENUM = "ENUM"
QUANTILE = "QUANTILE"
CONTINUOUS_ACTION = "CONTINUOUS_ACTION"
FEATURE_TYPES = (
    BINARY,
    PROBABILITY,
    CONTINUOUS,
    BOXCOX,
    ENUM,
    QUANTILE,
    CONTINUOUS_ACTION,
)

ROW_DELIM = "\n"
COLUMN_DELIM = ";"

DEFAULT_MAX_UNIQUE_ENUM = 100


def _is_probability(feature_values):
    return np.all(0 <= feature_values) and np.all(feature_values <= 1)


def _is_binary(feature_values):
    return np.all(np.logical_or(feature_values == 0, feature_values == 1)) or np.min(
        feature_values
    ) == np.max(feature_values)


def _is_continuous(feature_values):
    return True


def _is_enum(feature_values, enum_threshold):
    are_all_ints = np.vectorize(lambda val: float(val).is_integer())
    return (
        np.min(feature_values) >= 0
        and len(np.unique(feature_values))  # All values must be positive
        <= enum_threshold
        and np.all(are_all_ints(feature_values))
    )


def identify_type(values, enum_threshold=DEFAULT_MAX_UNIQUE_ENUM):
    if _is_binary(values):
        return BINARY
    elif _is_probability(values):
        return PROBABILITY
    elif _is_enum(values, enum_threshold):
        return ENUM
    elif _is_continuous(values):
        return CONTINUOUS
    else:
        assert False
