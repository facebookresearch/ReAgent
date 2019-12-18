#!/usr/bin/env python3

import abc
from typing import Type

import torch
from ml.rl.core.registry_meta import RegistryMeta
from ml.rl.parameters import NormalizationData


class ValueNetBuilder(metaclass=RegistryMeta):
    """
    Base class for value-network builder.
    """

    @classmethod
    @abc.abstractmethod
    def config_type(cls) -> Type:
        """
        Return the config type. Must be conforming to Flow python 3 type API
        """
        pass

    @abc.abstractmethod
    def build_value_network(
        self, state_normalization_parameters: NormalizationData
    ) -> torch.nn.Module:
        pass
