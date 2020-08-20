#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.tagged_union import TaggedUnion
from reagent.models.model_feature_config_provider import ModelFeatureConfigProvider
from reagent.reporting.result_registries import PublishingResult, ValidationResult
from reagent.reporting.training_reports import TrainingReport


if True:  # Register modules for unions
    import reagent.reporting.oss_training_reports  # noqa
    import reagent.core.result_types  # noqa

    if IS_FB_ENVIRONMENT:
        import reagent.reporting.fb.fb_training_reports  # noqa
        import reagent.fb.models.model_feature_config_builder  # noqa
        import reagent.core.fb.fb_result_types  # noqa
        import reagent.core.fb.fb_types  # noqa


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
