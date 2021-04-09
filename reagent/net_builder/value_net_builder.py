#!/usr/bin/env python3

import abc

import torch
from reagent.core.parameters import NormalizationData
from reagent.core.registry_meta import RegistryMeta


class ValueNetBuilder:
    """
    Base class for value-network builder.
    """

    @abc.abstractmethod
    def build_value_network(
        self, state_normalization_data: NormalizationData
    ) -> torch.nn.Module:
        pass
