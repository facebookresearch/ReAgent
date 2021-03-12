#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from datetime import datetime as RecurringPeriod  # noqa
from typing import Dict, List, Optional

# Triggering registration to registries
import reagent.core.result_types  # noqa
import reagent.reporting.training_reports  # noqa
import reagent.workflow.training_reports  # noqa
from reagent.core.dataclasses import dataclass, field
from reagent.core.result_registries import (
    PublishingResult,
    TrainingReport,
    ValidationResult,
)
from reagent.core.result_registries import (
    PublishingResult,
    TrainingReport,
    ValidationResult,
)
from reagent.core.tagged_union import TaggedUnion
from reagent.core.types import BaseDataClass
from reagent.models.model_feature_config_provider import ModelFeatureConfigProvider
from reagent.preprocessing.normalization import (
    DEFAULT_MAX_QUANTILE_SIZE,
    DEFAULT_MAX_UNIQUE_ENUM,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_QUANTILE_K2_THRESHOLD,
)


try:
    from reagent.fb.models.model_feature_config_builder import (  # noqa
        ConfigeratorModelFeatureConfigProvider,
    )
except ImportError:
    pass


ModuleNameToEntityId = Dict[str, int]


@dataclass
class Dataset:
    parquet_url: str


@dataclass
class TableSpec:
    table_name: str
    table_sample: Optional[float] = None
    eval_table_sample: Optional[float] = None


@dataclass
class RewardOptions:
    custom_reward_expression: Optional[str] = None
    metric_reward_values: Optional[Dict[str, float]] = None


@dataclass
class ReaderOptions:
    minibatch_size: int = 1024
    petastorm_reader_pool_type: str = "thread"


@dataclass
class ResourceOptions:
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
    allowedlist_features: Optional[List[int]] = None
    assert_allowedlist_feature_coverage: bool = True


@ModelFeatureConfigProvider.fill_union()
class ModelFeatureConfigProvider__Union(TaggedUnion):
    pass


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
    output_paths: Dict[str, str] = field(default_factory=dict)
    validation_result: Optional[ValidationResult__Union] = None
    publishing_result: Optional[PublishingResult__Union] = None
    training_report: Optional[RLTrainingReport] = None
