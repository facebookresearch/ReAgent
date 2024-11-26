#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

from datetime import datetime as RecurringPeriod  # noqa
from typing import Dict, List, Optional, Tuple

# Triggering registration to registries
# pyre-fixme[21]: Could not find module `reagent.core.result_types`.
import reagent.core.result_types  # noqa

# pyre-fixme[21]: Could not find module `reagent.workflow.training_reports`.
import reagent.workflow.training_reports  # noqa

# pyre-fixme[21]: Could not find module `reagent.core.dataclasses`.
from reagent.core.dataclasses import dataclass, field

# pyre-fixme[21]: Could not find module `reagent.core.result_registries`.
from reagent.core.result_registries import (
    PublishingResult,
    TrainingReport,
    ValidationResult,
)

# pyre-fixme[21]: Could not find module `reagent.core.tagged_union`.
from reagent.core.tagged_union import TaggedUnion

# pyre-fixme[21]: Could not find module `reagent.models.model_feature_config_provider`.
from reagent.models.model_feature_config_provider import ModelFeatureConfigProvider

# pyre-fixme[21]: Could not find module `reagent.preprocessing.normalization`.
from reagent.preprocessing.normalization import (
    DEFAULT_MAX_QUANTILE_SIZE,
    DEFAULT_MAX_UNIQUE_ENUM,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_QUANTILE_K2_THRESHOLD,
)


ModuleNameToEntityId = Dict[str, int]


@dataclass
class Dataset:
    # pyre-fixme[13]: Attribute `parquet_url` is never initialized.
    parquet_url: str


@dataclass
class TableSpec:
    # pyre-fixme[13]: Attribute `table_name` is never initialized.
    table_name: str
    table_sample: Optional[float] = None
    eval_table_sample: Optional[float] = None
    test_table_sample: Optional[float] = None


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
    gpu: int = 1

    @property
    def use_gpu(self) -> bool:
        return self.gpu > 0

    ## Below is for internal use
    cpu: Optional[int] = None
    # "-1" or "xxG" where "xx" is a positive integer
    memory: Optional[str] = "40g"
    min_nodes: Optional[int] = 1
    max_nodes: Optional[int] = 1


@dataclass
class PreprocessingOptions:
    # pyre-fixme[16]: Module `reagent` has no attribute `preprocessing`.
    num_samples: int = DEFAULT_NUM_SAMPLES
    # pyre-fixme[16]: Module `reagent` has no attribute `preprocessing`.
    max_unique_enum_values: int = DEFAULT_MAX_UNIQUE_ENUM
    # pyre-fixme[16]: Module `reagent` has no attribute `preprocessing`.
    quantile_size: int = DEFAULT_MAX_QUANTILE_SIZE
    # pyre-fixme[16]: Module `reagent` has no attribute `preprocessing`.
    quantile_k2_threshold: float = DEFAULT_QUANTILE_K2_THRESHOLD
    skip_box_cox: bool = False
    skip_quantiles: bool = True
    feature_overrides: Optional[Dict[int, str]] = None
    tablesample: Optional[float] = None
    set_missing_value_to_zero: Optional[bool] = False
    allowedlist_features: Optional[List[int]] = None
    assert_allowedlist_feature_coverage: bool = True


@ModelFeatureConfigProvider.fill_union()
# pyre-fixme[11]: Annotation `TaggedUnion` is not defined as a type.
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
    # pyre-fixme[16]: Module `reagent` has no attribute `core`.
    output_paths: Dict[str, str] = field(default_factory=dict)
    validation_result: Optional[ValidationResult__Union] = None
    publishing_result: Optional[PublishingResult__Union] = None
    training_report: Optional[RLTrainingReport] = None
    # pyre-fixme[16]: Module `reagent` has no attribute `core`.
    logger_data: Dict[str, Dict[str, List[Tuple[float, float]]]] = field(
        default_factory=dict
    )


@dataclass
class TrainerConf:
    pass
