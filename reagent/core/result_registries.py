#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from dataclasses import field
from typing import List, Any

from reagent.core.dataclasses import dataclass
from reagent.core.registry_meta import RegistryMeta


@dataclass
class TrainingReport(metaclass=RegistryMeta):
    plots: List[Any] = field(default_factory=list)


@dataclass
class PublishingResult(metaclass=RegistryMeta):
    success: bool


@dataclass
class ValidationResult(metaclass=RegistryMeta):
    should_publish: bool
