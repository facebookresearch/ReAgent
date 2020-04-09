#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from ml.rl.core.dataclasses import dataclass
from ml.rl.core.registry_meta import RegistryMeta


class TrainingReport(metaclass=RegistryMeta):
    pass


@dataclass
class PublishingResult(metaclass=RegistryMeta):
    success: bool


@dataclass
class ValidationResult(metaclass=RegistryMeta):
    should_publish: bool
