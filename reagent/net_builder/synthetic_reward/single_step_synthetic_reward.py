#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

from typing import List, Optional

import reagent.core.types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import NormalizationData, param_hash
from reagent.models.base import ModelBase
from reagent.models.synthetic_reward import (
    SingleStepSyntheticRewardNet,
    SyntheticRewardNet,
)
from reagent.net_builder.synthetic_reward_net_builder import SyntheticRewardNetBuilder
from reagent.preprocessing.normalization import get_num_output_features


@dataclass
class SingleStepSyntheticReward(SyntheticRewardNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    last_layer_activation: str = "sigmoid"
    use_batch_norm: bool = False
    use_layer_norm: bool = False

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
        net = SingleStepSyntheticRewardNet(
            state_dim=state_dim,
            action_dim=action_dim,
            sizes=self.sizes,
            activations=self.activations,
            last_layer_activation=self.last_layer_activation,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
        )
        return SyntheticRewardNet(net)
