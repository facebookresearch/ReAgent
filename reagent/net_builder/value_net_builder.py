#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import abc

import torch
from reagent.core.parameters import NormalizationData


class ValueNetBuilder:
    """
    Base class for value-network builder.
    """

    @abc.abstractmethod
    def build_value_network(
        self, state_normalization_data: NormalizationData
    ) -> torch.nn.Module:
        pass
