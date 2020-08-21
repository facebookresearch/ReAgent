#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from reagent.core.dataclasses import dataclass
from reagent.workflow.result_registries import PublishingResult, ValidationResult


@dataclass
class NoPublishingResults(PublishingResult):
    __registry_name__ = "no_publishing_results"


@dataclass
class NoValidationResults(ValidationResult):
    __registry_name__ = "no_validation_results"
