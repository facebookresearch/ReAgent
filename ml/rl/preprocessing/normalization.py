from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
from scipy import stats
from scipy.stats.mstats import mquantiles
import json
import numpy as np
import six

import logging
logger = logging.getLogger(__name__)

from ml.rl.preprocessing import identify_types
from ml.rl.preprocessing.identify_types import DEFAULT_MAX_UNIQUE_ENUM

NormalizationParameters = namedtuple(
    'NormalizationParameters',
    [
        'feature_type',
        'boxcox_lambda',
        'boxcox_shift',
        'mean',
        'stddev',
        'possible_values',  # Assume present for ENUM type and sorted
        'quantiles',  # Assume present for QUANTILE type and sorted
    ]
)

BOX_COX_MAX_STDDEV = 1e8
BOX_COX_MARGIN = 1e-4
MISSING_VALUE = -1337.1337
DEFAULT_QUANTILE_K2_THRESHOLD = 1000.0
MINIMUM_SAMPLES_TO_IDENTIFY = 20
DEFAULT_MAX_QUANTILE_SIZE = 20


def _identify_parameter(
    values, feature_type, quantile_size, quantile_k2_threshold
):
    boxcox_lambda = None
    boxcox_shift = 0
    mean = 0
    stddev = 1
    possible_values = None
    quantiles = None
    assert feature_type in [
        identify_types.CONTINUOUS, identify_types.PROBABILITY,
        identify_types.BINARY, identify_types.ENUM, identify_types.QUANTILE
    ], "unknown type {}".format(feature_type)
    assert len(
        values
    ) >= MINIMUM_SAMPLES_TO_IDENTIFY, "insufficient information to identify parameter"

    min_value = np.min(values)
    max_value = np.max(values)
    if feature_type == identify_types.CONTINUOUS:
        assert min_value < max_value, "Binary feature marked as continuous"
        k2_original, p_original = stats.normaltest(values)

        # shift can be estimated but not in scipy
        boxcox_shift = float(min_value * -1)
        candidate_values, lmbda = stats.boxcox(
            np.maximum(values + boxcox_shift, BOX_COX_MARGIN)
        )
        k2_boxcox, p_boxcox = stats.normaltest(candidate_values)
        logger.info(
            "Feature stats.  Original K2: {} P: {} Boxcox K2: {} P: {}".
            format(k2_original, p_original, k2_boxcox, p_boxcox)
        )
        if lmbda < 0.9 or lmbda > 1.1:
            # Lambda is far enough from 1.0 to be worth doing boxcox
            if k2_original > k2_boxcox * 10 and k2_boxcox <= quantile_k2_threshold:
                # The boxcox output is significantly more normally distributed
                # than the original data and is normal enough to apply
                # effectively.

                stddev = np.std(candidate_values, ddof=1)
                # Unclear whether this happens in practice or not
                if np.isfinite(stddev) and stddev < BOX_COX_MAX_STDDEV and \
                   not np.isclose(stddev, 0):
                    values = candidate_values
                    boxcox_lambda = float(lmbda)
        if boxcox_lambda is None:
            boxcox_shift = None
        if boxcox_lambda is None and k2_original > quantile_k2_threshold:
            feature_type = identify_types.QUANTILE
            quantiles = mquantiles(
                values,
                np.arange(quantile_size, dtype=np.float32) /
                float(quantile_size)
            ).astype(float).tolist()
            logger.info(
                "Feature is non-normal, using quantiles: {}".format(quantiles)
            )

    if feature_type == identify_types.CONTINUOUS:
        mean = float(np.mean(values))
        values = values - mean
        stddev = float(np.std(values, ddof=1))
        if np.isclose(stddev, 0) or not np.isfinite(stddev):
            stddev = 1
        values /= stddev

    if feature_type == identify_types.ENUM:
        possible_values = np.unique(values).astype(float).tolist()

    return NormalizationParameters(
        feature_type, boxcox_lambda, boxcox_shift, mean, stddev,
        possible_values, quantiles
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


def identify_parameters(
    feature_value_map,
    max_unique_enum_values=DEFAULT_MAX_UNIQUE_ENUM,
    quantile_size=DEFAULT_MAX_QUANTILE_SIZE,
    quantile_k2_threshold=DEFAULT_QUANTILE_K2_THRESHOLD,
):
    initial_feature_types = identify_types.identify_types(
        feature_value_map, max_unique_enum_values
    )
    parameters = {}
    for feature_name, feature_values in feature_value_map.items():
        if feature_values.shape[0] >= MINIMUM_SAMPLES_TO_IDENTIFY:
            logger.info("Identifying feature {}".format(feature_name))
            parameters[feature_name] = _identify_parameter(
                feature_values, initial_feature_types[feature_name],
                quantile_size, quantile_k2_threshold
            )
        else:
            logger.info("Feature {} has too few samples".format(feature_name))
    return parameters


def deserialize(parameters_json):
    parameters = {}
    for feature, feature_parameters in six.iteritems(parameters_json):
        parameters[feature] = NormalizationParameters(
            **json.loads(feature_parameters)
        )
    return parameters


def serialize(parameters):
    parameters_json = {}
    for feature, feature_parameters in six.iteritems(parameters):
        parameters_json[feature] = json.dumps(feature_parameters._asdict())
    return parameters_json
