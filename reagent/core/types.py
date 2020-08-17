#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from datetime import datetime as RecurringPeriod  # noqa
from typing import Dict, List, Optional

# Triggering registration to registries
import reagent.core.result_types  # noqa
import reagent.reporting.oss_training_reports  # noqa
from reagent.core.dataclasses import dataclass
from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.tagged_union import TaggedUnion  # noqa F401
from reagent.models.model_feature_config_provider import ModelFeatureConfigProvider
from reagent.preprocessing.normalization import (
    DEFAULT_MAX_QUANTILE_SIZE,
    DEFAULT_MAX_UNIQUE_ENUM,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_QUANTILE_K2_THRESHOLD,
)
from reagent.reporting.result_registries import PublishingResult, ValidationResult
from reagent.reporting.training_reports import TrainingReport
from reagent.types import BaseDataClass


@dataclass
class Dataset:
    pass


@dataclass
class OssDataset(Dataset):
    parquet_url: str


@dataclass
class TableSpec:
    table: str
    table_sample: Optional[float] = None
    eval_table_sample: Optional[float] = None


@dataclass
class RewardOptions:
    custom_reward_expression: Optional[str] = None
    metric_reward_values: Optional[Dict[str, float]] = None


@dataclass
class ReaderOptions:
    pass


@dataclass
class OssReaderOptions(ReaderOptions):
    petastorm_reader_pool_type: str = "thread"


@dataclass
class ResourceOptions:
    pass


@dataclass
class VarianceThreshold:
    avg: float = 1.0
    var: float = 10.0
    non_zero_ratio: float = 1.0


IGNORE_SANITY_CHECK_FAILURE = True


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
    variance_threshold: VarianceThreshold = VarianceThreshold()
    sequence_feature_id: Optional[int] = None

    ignore_sanity_check_failure: bool = IGNORE_SANITY_CHECK_FAILURE
    ignore_sanity_check_task: bool = False
    load_from_operator_id: Optional[int] = None
    skip_sanity_check: bool = False

    # IdMappings are stored in manifold folder:
    # "tree/{namespace}/{tablename}/{ds}/{base_mapping_name}/{embedding_table_name}"
    base_mapping_name: str = "DefaultMappingName"

    ### below here for preprocessing sparse features ###
    # If the number of occurrences of any raw features ids is lower than this, we
    # ignore those feature ids when constructing the IdMapping
    sparse_threshold: int = 0


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
class TrainingReport__Union(TaggedUnion):
    pass


@dataclass
class RLTrainingOutput:
    validation_result: Optional[ValidationResult__Union] = None
    publishing_result: Optional[PublishingResult__Union] = None
    training_report: Optional[TrainingReport__Union] = None
    local_output_path: Optional[str] = None
