from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import six

BINARY = "BINARY"
PROBABILITY = "PROBABILITY"
CONTINUOUS = "CONTINUOUS"
ENUM = "ENUM"

ROW_DELIM = '\n'
COLUMN_DELIM = ';'

DEFAULT_MAX_UNIQUE_ENUM = 1000


def _is_probability(feature_values):
    return np.all(0 <= feature_values) and np.all(feature_values <= 1)


def _is_binary(feature_values):
    return np.all(np.logical_or(feature_values == 0, feature_values == 1)) \
        or np.min(feature_values) == np.max(feature_values)


def _is_continuous(feature_values):
    return True


def _is_enum(feature_values, enum_threshold):
    are_all_ints = np.vectorize(lambda val: float(val).is_integer())
    return (
        float(len(np.unique(feature_values))) < enum_threshold and
        np.all(are_all_ints(feature_values))
    )


def identify_types(feature_values, enum_threshold=DEFAULT_MAX_UNIQUE_ENUM):
    result = {}
    for feature_name, values in six.iteritems(feature_values):
        if _is_binary(values):
            result[feature_name] = BINARY
        elif _is_probability(values):
            result[feature_name] = PROBABILITY
        elif _is_enum(values, enum_threshold):
            result[feature_name] = ENUM
        elif _is_continuous(values):
            result[feature_name] = CONTINUOUS
        else:
            assert False
    return result


def identify_types_dict(feature_values, enum_threshold=DEFAULT_MAX_UNIQUE_ENUM):
    types = identify_types(feature_values, enum_threshold)
    return {feature_name: types[feature_name] for feature_name in feature_values}
