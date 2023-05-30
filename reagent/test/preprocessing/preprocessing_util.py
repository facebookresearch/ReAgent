#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
from scipy import stats


BINARY_FEATURE_ID = 1
BINARY_FEATURE_ID_2 = 2
BOXCOX_FEATURE_ID = 3
CONTINUOUS_FEATURE_ID = 4
CONTINUOUS_FEATURE_ID_2 = 5
ENUM_FEATURE_ID = 6
PROBABILITY_FEATURE_ID = 7
QUANTILE_FEATURE_ID = 8
CONTINUOUS_ACTION_FEATURE_ID = 9
CONTINUOUS_ACTION_FEATURE_ID_2 = 10
CONTINUOUS_SMALL_FEATURE_ID = 11


def id_to_type(id) -> str:
    if id == BINARY_FEATURE_ID or id == BINARY_FEATURE_ID_2:
        return "BINARY"
    if id == BOXCOX_FEATURE_ID:
        return "BOXCOX"
    if id == CONTINUOUS_FEATURE_ID or id == CONTINUOUS_FEATURE_ID_2:
        return "CONTINUOUS"
    if id == ENUM_FEATURE_ID:
        return "ENUM"
    if id == PROBABILITY_FEATURE_ID:
        return "PROBABILITY"
    if id == QUANTILE_FEATURE_ID:
        return "QUANTILE"
    if id == CONTINUOUS_ACTION_FEATURE_ID or id == CONTINUOUS_ACTION_FEATURE_ID_2:
        return "CONTINUOUS_ACTION"
    if id == CONTINUOUS_SMALL_FEATURE_ID:
        return "DO_NOT_PREPROCESS"
    # pyre-fixme[58]: `+` is not supported for operand types `str` and `(object) ->
    #  int`.
    assert False, "Invalid feature id: " + id


def read_data():
    np.random.seed(1)
    feature_value_map = {}
    feature_value_map[BINARY_FEATURE_ID] = stats.bernoulli.rvs(0.5, size=10000).astype(
        np.float32
    )
    feature_value_map[BINARY_FEATURE_ID_2] = stats.bernoulli.rvs(
        0.5, size=10000
    ).astype(np.float32)
    feature_value_map[CONTINUOUS_FEATURE_ID] = stats.norm.rvs(size=10000).astype(
        np.float32
    )
    feature_value_map[CONTINUOUS_FEATURE_ID_2] = stats.norm.rvs(size=10000).astype(
        np.float32
    )
    feature_value_map[BOXCOX_FEATURE_ID] = stats.expon.rvs(size=10000).astype(
        np.float32
    )
    feature_value_map[ENUM_FEATURE_ID] = (
        stats.randint.rvs(0, 10, size=10000) * 1000
    ).astype(np.float32)
    feature_value_map[QUANTILE_FEATURE_ID] = np.concatenate(
        (stats.norm.rvs(size=5000), stats.expon.rvs(size=5000))
    ).astype(np.float32)
    feature_value_map[PROBABILITY_FEATURE_ID] = np.clip(
        stats.beta.rvs(a=2.0, b=2.0, size=10000).astype(np.float32), 0.01, 0.99
    )
    feature_value_map[CONTINUOUS_ACTION_FEATURE_ID] = stats.norm.rvs(size=10000).astype(
        np.float32
    )
    feature_value_map[CONTINUOUS_ACTION_FEATURE_ID_2] = stats.norm.rvs(
        size=10000
    ).astype(np.float32)
    feature_value_map[CONTINUOUS_SMALL_FEATURE_ID] = np.clip(
        stats.norm.rvs(scale=1e-6, size=10000).astype(np.float32), -1e-5, 1e-5
    )

    return feature_value_map
