#!/usr/bin/env python3

import abc

import torch
from reagent.core.registry_meta import RegistryMeta
from reagent.parameters import NormalizationData


class ValueNetBuilder(metaclass=RegistryMeta):
    """
    Base class for value-network builder.
    """

    @abc.abstractmethod
    def build_value_network(
        self, state_normalization_data: NormalizationData
    ) -> torch.nn.Module:
        pass
