#!/usr/bin/env python3

import json
import logging
from collections import namedtuple

import numpy as np
import six
from ml.rl.preprocessing import identify_types
from ml.rl.preprocessing.identify_types import DEFAULT_MAX_UNIQUE_ENUM
from scipy import stats
from scipy.stats.mstats import mquantiles


logger = logging.getLogger(__name__)


NormalizationParameters = namedtuple(
    "NormalizationParameters",
    [
        "feature_type",
        "boxcox_lambda",
        "boxcox_shift",
        "mean",
        "stddev",
        "possible_values",  # Assume present for ENUM type
        "quantiles",  # Assume present for QUANTILE type and sorted
        "min_value",
        "max_value",
    ],
)
NormalizationParameters.__new__.__defaults__ = (None,) * len(
    NormalizationParameters._fields
)

BOX_COX_MAX_STDDEV = 1e8
BOX_COX_MARGIN = 1e-4
MISSING_VALUE = -1337.1337
DEFAULT_QUANTILE_K2_THRESHOLD = 1000.0
MINIMUM_SAMPLES_TO_IDENTIFY = 20
DEFAULT_MAX_QUANTILE_SIZE = 20


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def identify_parameter(
    values,
    max_unique_enum_values=DEFAULT_MAX_UNIQUE_ENUM,
    quantile_size=DEFAULT_MAX_QUANTILE_SIZE,
    quantile_k2_threshold=DEFAULT_QUANTILE_K2_THRESHOLD,
    skip_box_cox=False,
    skip_quantiles=False,
    feature_type=None,
):
    if feature_type is None:
        feature_type = identify_types.identify_type(values, max_unique_enum_values)

    boxcox_lambda = None
    boxcox_shift = 0
    mean = 0
    stddev = 1
    possible_values = None
    quantiles = None
    assert feature_type in [
        identify_types.CONTINUOUS,
        identify_types.PROBABILITY,
        identify_types.BINARY,
        identify_types.ENUM,
    ], "unknown type {}".format(feature_type)
    assert (
        len(values) >= MINIMUM_SAMPLES_TO_IDENTIFY
    ), "insufficient information to identify parameter"

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
            "Feature stats.  Original K2: {} P: {} Boxcox K2: {} P: {}".format(
                k2_original, p_original, k2_boxcox, p_boxcox
            )
        )
        if lmbda < 0.9 or lmbda > 1.1:
            # Lambda is far enough from 1.0 to be worth doing boxcox
            if k2_original > k2_boxcox * 10 and k2_boxcox <= quantile_k2_threshold:
                # The boxcox output is significantly more normally distributed
                # than the original data and is normal enough to apply
                # effectively.

                stddev = np.std(candidate_values, ddof=1)
                # Unclear whether this happens in practice or not
                if (
                    np.isfinite(stddev)
                    and stddev < BOX_COX_MAX_STDDEV
                    and not np.isclose(stddev, 0)
                ):
                    values = candidate_values
                    boxcox_lambda = float(lmbda)
        if boxcox_lambda is None or skip_box_cox:
            boxcox_shift = None
            boxcox_lambda = None
        if boxcox_lambda is not None:
            feature_type = identify_types.BOXCOX
        if (
            boxcox_lambda is None
            and k2_original > quantile_k2_threshold
            and (not skip_quantiles)
        ):
            feature_type = identify_types.QUANTILE
            quantiles = (
                np.unique(
                    mquantiles(
                        values,
                        np.arange(quantile_size + 1, dtype=np.float64)
                        / float(quantile_size),
                        alphap=0.0,
                        betap=1.0,
                    )
                )
                .astype(float)
                .tolist()
            )
            logger.info("Feature is non-normal, using quantiles: {}".format(quantiles))

    if (
        feature_type == identify_types.CONTINUOUS
        or feature_type == identify_types.BOXCOX
    ):
        mean = float(np.mean(values))
        values = values - mean
        stddev = float(np.std(values, ddof=1))
        if np.isclose(stddev, 0) or not np.isfinite(stddev):
            stddev = 1
        values /= stddev

    if feature_type == identify_types.ENUM:
        possible_values = np.unique(values.astype(int)).tolist()

    return NormalizationParameters(
        feature_type,
        boxcox_lambda,
        boxcox_shift,
        mean,
        stddev,
        possible_values,
        quantiles,
        min_value,
        max_value,
    )


def get_num_output_features(normalization_parmeters):
    return sum(
        map(
            lambda np: (
                len(np.possible_values) if np.feature_type == identify_types.ENUM else 1
            ),
            normalization_parmeters.values(),
        )
    )


def deserialize(parameters_json):
    parameters = {}
    for feature, feature_parameters in six.iteritems(parameters_json):
        params = NormalizationParameters(**json.loads(feature_parameters))
        # Check for negative enum IDs
        if params.feature_type == identify_types.ENUM:
            for x in params.possible_values:
                if x < 0:
                    logger.fatal(
                        "Invalid enum ID: "
                        + str(x)
                        + " in feature: "
                        + feature
                        + " with possible_values "
                        + str(params.possible_values)
                        + " (raw: "
                        + feature_parameters
                        + ")"
                    )
                    raise Exception("Invalid enum ID")
        parameters[feature] = params
    return parameters


def serialize_one(feature_parameters):
    return json.dumps(feature_parameters._asdict(), cls=NumpyEncoder)


def serialize(parameters):
    parameters_json = {}
    for feature, feature_parameters in six.iteritems(parameters):
        parameters_json[feature] = serialize_one(feature_parameters)
    return parameters_json
