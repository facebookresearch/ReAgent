#!/usr/bin/env python3

import numpy as np
from ml.rl.preprocessing.identify_types import (
    BINARY,
    BOXCOX,
    CONTINUOUS,
    ENUM,
    PROBABILITY,
    QUANTILE,
)
from scipy import stats


def read_data():
    np.random.seed(1)
    feature_value_map = {}
    feature_value_map[BINARY] = stats.bernoulli.rvs(0.5, size=10000).astype(np.float32)
    feature_value_map[BINARY + "_2"] = stats.bernoulli.rvs(0.5, size=10000).astype(
        np.float32
    )
    feature_value_map[CONTINUOUS] = stats.norm.rvs(size=10000).astype(np.float32)
    feature_value_map[CONTINUOUS + "_2"] = stats.norm.rvs(size=10000).astype(np.float32)
    feature_value_map[BOXCOX] = stats.expon.rvs(size=10000).astype(np.float32)
    feature_value_map[ENUM] = (stats.randint.rvs(0, 10, size=10000) * 1000).astype(
        np.float32
    )
    feature_value_map[QUANTILE] = np.concatenate(
        (stats.norm.rvs(size=5000), stats.expon.rvs(size=5000))
    ).astype(np.float32)
    feature_value_map[PROBABILITY] = np.clip(
        stats.beta.rvs(a=2.0, b=2.0, size=10000).astype(np.float32), -4, 4
    )

    features = [
        BINARY,
        BINARY + "_2",
        PROBABILITY,
        CONTINUOUS,
        CONTINUOUS + "_2",
        BOXCOX,
        ENUM,
        QUANTILE,
    ]

    return features, feature_value_map
