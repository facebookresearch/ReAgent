#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import List, Optional

import reagent.core.types as rlt
from reagent.core.dataclasses import dataclass
from reagent.core.parameters import NormalizationData, param_hash
from reagent.models.base import ModelBase
from reagent.models.synthetic_reward import (
    TransformerSyntheticRewardNet,
    SyntheticRewardNet,
)
from reagent.net_builder.synthetic_reward_net_builder import SyntheticRewardNetBuilder
from reagent.preprocessing.normalization import get_num_output_features


@dataclass
class TransformerSyntheticReward(SyntheticRewardNetBuilder):
    __hash__ = param_hash

    nhead: int = 1
    d_model: int = 128
    num_encoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.0
    activation: str = "relu"
    last_layer_activation: str = "leaky_relu"
    layer_norm_eps: float = 1e-5
    max_len: int = 10

    def build_synthetic_reward_network(
        self,
        state_normalization_data: NormalizationData,
        action_normalization_data: Optional[NormalizationData] = None,
        discrete_action_names: Optional[List[str]] = None,
        state_feature_config: Optional[rlt.ModelFeatureConfig] = None,
        action_feature_config: Optional[rlt.ModelFeatureConfig] = None,
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

        net = TransformerSyntheticRewardNet(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            last_layer_activation=self.last_layer_activation,
            layer_norm_eps=self.layer_norm_eps,
            max_len=self.max_len,
        )
        return SyntheticRewardNet(net=net)
