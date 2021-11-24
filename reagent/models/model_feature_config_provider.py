#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import abc

import reagent.core.types as rlt
from reagent.core.dataclasses import dataclass
from reagent.core.registry_meta import RegistryMeta


class ModelFeatureConfigProvider(metaclass=RegistryMeta):
    @abc.abstractmethod
    def get_model_feature_config(self) -> rlt.ModelFeatureConfig:
        pass


@dataclass
class RawModelFeatureConfigProvider(ModelFeatureConfigProvider, rlt.ModelFeatureConfig):
    __registry_name__ = "raw"

    def get_model_feature_config(self) -> rlt.ModelFeatureConfig:
        return self
