#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from dataclasses import dataclass
from typing import Optional

from reagent.core.union import (
    PublishingResult__Union,
    TrainingReport__Union,
    ValidationResult__Union,
)


@dataclass
class RLTrainingOutput:
    validation_result: Optional[ValidationResult__Union] = None
    publishing_result: Optional[PublishingResult__Union] = None
    training_report: Optional[TrainingReport__Union] = None
    local_output_path: Optional[str] = None
