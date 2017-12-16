from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
from scipy import stats
import json
import numpy as np
import six

from ml.rl.preprocessing import identify_types

NormalizationParameters = namedtuple(
    'NormalizationParameters',
    [
        'feature_type', 'boxcox_lambda', 'boxcox_shift', 'mean', 'stddev',
        'possible_values'  # assumes this is present for ENUM type and is sorted
    ]
)

BOX_COX_MAX_STDDEV = 1e8
BOX_COX_MIN_FRACTION = 1e-4
BOX_COX_MIN_VALUE = 1e-4
MISSING_VALUE = -1337.1337


def _identify_parameter(values, feature_type):
    boxcox_lambda = None
    boxcox_shift = 0
    mean = 0
    stddev = 1
    assert feature_type == identify_types.CONTINUOUS or \
        feature_type == identify_types.PROBABILITY or \
        feature_type == identify_types.BINARY,\
        "unknown type {}".format(feature_type)
    min_value = np.min(values)
    max_value = np.max(values)
    if feature_type == identify_types.CONTINUOUS:
        assert min_value < max_value, "Binary feature marked as continuous"
        # shift can be estimated but not in scipy
        boxcox_shift = min_value - \
            abs(min_value) * BOX_COX_MIN_FRACTION - BOX_COX_MIN_VALUE
        candidate_values, lmbda = stats.boxcox(values - boxcox_shift)
        stddev = np.std(candidate_values, ddof=1)
        # Unclear whether this happens in practice or not
        if np.isfinite(stddev) and stddev < BOX_COX_MAX_STDDEV and \
           not np.isclose(stddev, 0):
            values = candidate_values
            boxcox_lambda = lmbda

    if feature_type != identify_types.BINARY:
        mean = np.mean(values)
        values = values - mean
        stddev = np.std(values, ddof=1)
        if np.isclose(stddev, 0) or not np.isfinite(stddev):
            stddev = 1
        values /= stddev
    return NormalizationParameters(
        feature_type, boxcox_lambda, boxcox_shift, mean, stddev, None
    )


def get_num_output_features(normalization_parmeters):
    return sum(
        map(
            lambda np: (
                len(np.possible_values) if np.feature_type == identify_types.ENUM
                else 1
            ),
            normalization_parmeters.values()
        )
    )


def identify_parameters(feature_values, types):
    parameters = {}
    for feature_name in feature_values:
        parameters[feature_name] = _identify_parameter(
            feature_values[feature_name], types[feature_name]
        )
    return parameters


def write_parameters(f, parameters):
    json.dump(
        {
            feature_name: parameters[feature_name]._asdict()
            for feature_name in parameters
        }, f
    )


def load_parameters(f):
    parameter_map = json.load(f)
    parameters = {}
    for feature, feature_parameters in six.iteritems(parameter_map):
        if 'possible_values' not in feature_parameters:
            feature_parameters['possible_values'] = None
        parameters[feature] = NormalizationParameters(**feature_parameters)
    return parameters
