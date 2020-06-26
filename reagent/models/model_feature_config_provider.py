#!/usr/bin/env python3

import abc

import reagent.types as rlt
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
