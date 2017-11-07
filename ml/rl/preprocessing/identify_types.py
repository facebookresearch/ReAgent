from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import six

BINARY = "BINARY"
PROBABILITY = "PROBABILITY"
CONTINUOUS = "CONTINUOUS"

ROW_DELIM = '\n'
COLUMN_DELIM = ';'


def _is_probability(feature_values):
    return np.all(0 <= feature_values) and np.all(feature_values <= 1)


def _is_binary(feature_values):
    return np.all(np.logical_or(feature_values == 0, feature_values == 1)) \
        or np.min(feature_values) == np.max(feature_values)


def _is_continuous(feature_values):
    return True


def identify_types(feature_values):
    result = {}
    for feature_name, values in six.iteritems(feature_values):
        if _is_binary(values):
            result[feature_name] = BINARY
        elif _is_probability(values):
            result[feature_name] = PROBABILITY
        elif _is_continuous(values):
            result[feature_name] = CONTINUOUS
        else:
            assert False
    return result


def identify_types_dict(feature_values):
    types = identify_types(feature_values)
    return {feature_name: types[feature_name] for feature_name in feature_values}
