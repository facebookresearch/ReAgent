#!/usr/bin/env python3

from typing import List, Optional

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import NormalizationData, param_hash, ConvNetParameters
from reagent.models.base import ModelBase
from reagent.models.synthetic_reward import NGramSyntheticRewardNet
from reagent.net_builder.synthetic_reward_net_builder import SyntheticRewardNetBuilder
from reagent.preprocessing.normalization import get_num_output_features


@dataclass
class NGramSyntheticReward(SyntheticRewardNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    last_layer_activation: str = "sigmoid"
    context_size: int = 3
    conv_net_params: Optional[ConvNetParameters] = None

    def build_synthetic_reward_network(
        self,
        state_normalization_data: NormalizationData,
        action_normalization_data: Optional[NormalizationData] = None,
        discrete_action_names: Optional[List[str]] = None,
    ) -> ModelBase:
        state_dim = get_num_output_features(
            state_normalization_data.dense_normalization_parameters
        )
        if not discrete_action_names:
            assert action_normalization_data is not None
            action_dim = get_num_output_features(
                action_normalization_data.dense_normalization_parameters
            )
        else:
            action_dim = len(discrete_action_names)
        return NGramSyntheticRewardNet(
            state_dim=state_dim,
            action_dim=action_dim,
            sizes=self.sizes,
            activations=self.activations,
            last_layer_activation=self.last_layer_activation,
            context_size=self.context_size,
            conv_net_params=self.conv_net_params,
        )

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
        raise NotImplementedError(
            "N-gram Synthetic Reward Predictor has not been implemented"
        )
