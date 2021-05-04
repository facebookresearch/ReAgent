#!/usr/bin/env python3

import abc
from typing import List, Optional

import torch
from reagent.core.parameters import NormalizationData
from reagent.models.base import ModelBase


class SyntheticRewardNetBuilder:
    """
    Base class for Synthetic Reward net builder.
    """

    @abc.abstractmethod
    def build_synthetic_reward_network(
        self,
        state_normalization_data: NormalizationData,
        action_normalization_data: Optional[NormalizationData] = None,
        discrete_action_names: Optional[List[str]] = None,
    ) -> ModelBase:
        pass

    def build_serving_module(
        self,
        synthetic_reward_network: ModelBase,
        state_normalization_data: NormalizationData,
        action_normalization_data: Optional[NormalizationData] = None,
        discrete_action_names: Optional[List[str]] = None,
    ) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        pass
