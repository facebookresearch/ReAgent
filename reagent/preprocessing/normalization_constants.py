#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from reagent.preprocessing.identify_types import (  # noqa
    DEFAULT_MAX_UNIQUE_ENUM,
    FEATURE_TYPES,
)


BOX_COX_MAX_STDDEV = 1e8
BOX_COX_MARGIN = 1e-4
MISSING_VALUE = -1337.1337
DEFAULT_QUANTILE_K2_THRESHOLD = 1000.0
MINIMUM_SAMPLES_TO_IDENTIFY = 20
DEFAULT_MAX_QUANTILE_SIZE = 20
DEFAULT_NUM_SAMPLES = 100000
MAX_FEATURE_VALUE = 6.0
MIN_FEATURE_VALUE = MAX_FEATURE_VALUE * -1
EPS = 1e-6
