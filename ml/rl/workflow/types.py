#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from datetime import datetime as RecurringPeriod  # noqa
from typing import Dict, List, NamedTuple, Optional

# Triggering registration to registries
import ml.rl.workflow.result_types  # noqa
import ml.rl.workflow.training_reports  # noqa
from ml.rl.core.dataclasses import dataclass
from ml.rl.preprocessing.normalization import (
    DEFAULT_MAX_QUANTILE_SIZE,
    DEFAULT_MAX_UNIQUE_ENUM,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_QUANTILE_K2_THRESHOLD,
)
from ml.rl.types import BaseDataClass
from ml.rl.workflow.result_registries import (
    PublishingResult,
    TrainingReport,
    ValidationResult,
)
from ml.rl.workflow.tagged_union import TaggedUnion  # noqa F401


@dataclass
class TableSpec(BaseDataClass):
    table_name: str


@dataclass
class Dataset:
    url: str


@dataclass
class RewardOptions:
    pass


@dataclass
class ReaderOptions:
    pass


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


@PublishingResult.fill_union()
class PublishingResult__Union(TaggedUnion):
    pass


@ValidationResult.fill_union()
class ValidationResult__Union(TaggedUnion):
    pass


@TrainingReport.fill_union()
class RLTrainingReport(TaggedUnion):
    pass


@dataclass
class RLTrainingOutput:
    validation_result: Optional[ValidationResult__Union] = None
    publishing_result: Optional[PublishingResult__Union] = None
    training_report: Optional[RLTrainingReport] = None
