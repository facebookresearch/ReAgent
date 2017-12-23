from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy import stats

from ml.rl.preprocessing.identify_types import BINARY, QUANTILE, ENUM, PROBABILITY


def read_data():
    np.random.seed(1)
    feature_value_map = {}
    feature_value_map[BINARY] = stats.bernoulli.rvs(
        0.5, size=10000
    ).astype(np.float32)
    feature_value_map['normal'] = stats.norm.rvs(size=10000).astype(np.float32)
    feature_value_map['boxcox'] = stats.expon.rvs(size=10000).astype(np.float32)
    feature_value_map[ENUM] = (stats.randint.rvs(0, 10, size=10000) *
                               1000).astype(np.float32)
    feature_value_map[QUANTILE] = np.concatenate(
        (stats.norm.rvs(size=10000), stats.expon.rvs(size=10000))
    ).astype(np.float32)
    feature_value_map[PROBABILITY] = stats.beta.rvs(
        a=2.0, b=2.0, size=10000
    ).astype(np.float32)

    return feature_value_map
