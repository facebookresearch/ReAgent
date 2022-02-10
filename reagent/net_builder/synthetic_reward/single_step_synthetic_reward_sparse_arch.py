#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import List, Optional

import reagent.core.types as rlt
import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import NormalizationData, param_hash
from reagent.core.torchrec_types import EmbeddingBagConfig, EmbeddingBagCollection
from reagent.core.utils import embedding_bag_configs_from_feature_configs
from reagent.models.base import ModelBase
from reagent.models.synthetic_reward_sparse_arch import (
    SingleStepSyntheticSparseArchRewardNet,
    SyntheticRewardSparseArchNet,
)
from reagent.net_builder.synthetic_reward_net_builder import SyntheticRewardNetBuilder
from reagent.preprocessing.normalization import get_num_output_features


@dataclass
class SingleStepSparseArchSyntheticReward(SyntheticRewardNetBuilder):
    __hash__ = param_hash

    dense_sizes: List[int] = field(default_factory=lambda: [256, 128])
    dense_activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    overall_sizes: List[int] = field(default_factory=lambda: [128, 1])
    overall_activations: List[str] = field(default_factory=lambda: ["relu", "sigmoid"])
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
        # Sparse features will be read from state_feature_config/action_feature_config
        feature_config_list: List[rlt.ModelFeatureConfig] = []
        assert state_feature_config is not None
        feature_config_list.append(state_feature_config)
        if discrete_action_names is None:
            assert action_feature_config is not None
            feature_config_list.append(action_feature_config)

        state_dense_dim = get_num_output_features(
            state_normalization_data.dense_normalization_parameters
        )
        if not discrete_action_names:
            assert action_normalization_data is not None
            action_dense_dim = get_num_output_features(
                action_normalization_data.dense_normalization_parameters
            )
        else:
            action_dense_dim = len(discrete_action_names)

        embedding_bag_configs: List[
            EmbeddingBagConfig
        ] = embedding_bag_configs_from_feature_configs(
            feature_config_list,
        )
        embedding_bag_col = EmbeddingBagCollection(
            device=torch.device("meta"), tables=embedding_bag_configs
        )
        net = SingleStepSyntheticSparseArchRewardNet(
            state_dense_dim=state_dense_dim,
            action_dense_dim=action_dense_dim,
            dense_sizes=self.dense_sizes,
            dense_activations=self.dense_activations,
            overall_sizes=self.overall_sizes,
            overall_activations=self.overall_activations,
            embedding_bag_collection=embedding_bag_col,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
        )
        return SyntheticRewardSparseArchNet(net)
