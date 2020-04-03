#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from dataclasses import dataclass
from datetime import datetime as RecurringPeriod  # noqa
from typing import Dict, List, NamedTuple, Optional

from ml.rl.polyfill.types_lib.union import TaggedUnion  # noqa F401
from ml.rl.preprocessing.normalization import (
    DEFAULT_MAX_QUANTILE_SIZE,
    DEFAULT_MAX_UNIQUE_ENUM,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_QUANTILE_K2_THRESHOLD,
)
from ml.rl.types import BaseDataClass
from ml.rl.workflow.result_types import PublishingResults, ValidationResults


@dataclass
class TableSpec(BaseDataClass):
    table_name: str


@dataclass
class PreprocessingOptions(BaseDataClass):
    num_samples: int = DEFAULT_NUM_SAMPLES
    max_unique_enum_values: int = DEFAULT_MAX_UNIQUE_ENUM
    quantile_size: int = DEFAULT_MAX_QUANTILE_SIZE
    quantile_k2_threshold: float = DEFAULT_QUANTILE_K2_THRESHOLD
    skip_box_cox: bool = False
    skip_quantiles: bool = True
    feature_overrides: Optional[Dict[int, str]] = None
    tablesample: Optional[float] = None
    set_missing_value_to_zero: Optional[bool] = False
    whitelist_features: Optional[List[int]] = None
    assert_whitelist_feature_coverage: bool = True


class PublishingOutput(NamedTuple):
    success: bool
    results: PublishingResults


class ValidationOutput(NamedTuple):
    should_publish: bool
    results: ValidationResults
