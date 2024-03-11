#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import collections
import logging

import numpy as np
from reagent.core.parameters import NormalizationParameters


logger = logging.getLogger(__name__)


def normalizer_helper(feats, feature_type, min_value=None, max_value=None):
    assert feature_type in (
        "DISCRETE_ACTION",
        "CONTINUOUS",
        "CONTINUOUS_ACTION",
    ), f"invalid feature type: {feature_type}."
    assert type(min_value) == type(max_value) and type(min_value) in (
        int,
        float,
        list,
        np.ndarray,
        type(None),
    ), f"invalid {type(min_value)}, {type(max_value)}"
    if type(min_value) in [int, float, type(None)]:
        min_value = [min_value] * len(feats)
        max_value = [max_value] * len(feats)
    normalization = collections.OrderedDict(
        [
            (
                feats[i],
                NormalizationParameters(
                    feature_type=feature_type,
                    boxcox_lambda=None,
                    boxcox_shift=None,
                    mean=0,
                    stddev=1,
                    possible_values=None,
                    quantiles=None,
                    min_value=float(min_value[i]) if min_value[i] is not None else None,
                    max_value=float(max_value[i]) if max_value[i] is not None else None,
                ),
            )
            for i in range(len(feats))
        ]
    )
    return normalization


def discrete_action_normalizer(feats):
    return normalizer_helper(feats, "DISCRETE_ACTION")


def only_continuous_normalizer(feats, min_value=None, max_value=None):
    return normalizer_helper(feats, "CONTINUOUS", min_value, max_value)


def only_continuous_action_normalizer(feats, min_value=None, max_value=None):
    return normalizer_helper(feats, "CONTINUOUS_ACTION", min_value, max_value)
