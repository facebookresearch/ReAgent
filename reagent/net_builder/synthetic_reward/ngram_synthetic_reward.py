#!/usr/bin/env python3

from typing import List, Optional

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import NormalizationData, param_hash, ConvNetParameters
from reagent.models.base import ModelBase
from reagent.models.synthetic_reward import (
    NGramConvolutionalNetwork,
    SyntheticRewardNet,
    NGramFullyConnectedNetwork,
)
from reagent.net_builder.synthetic_reward_net_builder import SyntheticRewardNetBuilder
from reagent.preprocessing.normalization import get_num_output_features


@dataclass
class NGramSyntheticReward(SyntheticRewardNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    last_layer_activation: str = "sigmoid"
    context_size: int = 3
    use_layer_norm: bool = False

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

        net = NGramFullyConnectedNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            sizes=self.sizes,
            activations=self.activations,
            last_layer_activation=self.last_layer_activation,
            context_size=self.context_size,
            use_layer_norm=self.use_layer_norm,
        )
        return SyntheticRewardNet(net)


@dataclass
class NGramConvNetSyntheticReward(SyntheticRewardNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    last_layer_activation: str = "sigmoid"
    context_size: int = 3
    conv_net_params: ConvNetParameters = field(
        default_factory=lambda: ConvNetParameters(
            conv_dims=[256, 128],
            conv_height_kernels=[1, 1],
            pool_types=["max", "max"],
            pool_kernel_sizes=[1, 1],
        )
    )
    use_layer_norm: bool = False

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

        net = NGramConvolutionalNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            sizes=self.sizes,
            activations=self.activations,
            last_layer_activation=self.last_layer_activation,
            context_size=self.context_size,
            conv_net_params=self.conv_net_params,
            use_layer_norm=self.use_layer_norm,
        )
        return SyntheticRewardNet(net)
