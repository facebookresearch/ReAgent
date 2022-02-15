#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Dict, List

import torch
from reagent.core import types as rlt
from reagent.core.torchrec_types import EmbeddingBagConfig
from reagent.core.utils import embedding_bag_configs_from_feature_configs
from reagent.models.base import ModelBase


class EmbeddingBagConcat(ModelBase):
    """
    Concatenating embedding with float features before passing the input
    to DQN
    """

    def __init__(
        self,
        state_dense_dim: int,
        model_feature_config: rlt.ModelFeatureConfig,
    ) -> None:
        super().__init__()
        assert state_dense_dim > 0, "state_dense_dim must be > 0, got {}".format(
            state_dense_dim
        )
        self.state_dense_dim = state_dense_dim
        # for input prototype
        self._id_list_feature_names: List[str] = [
            config.name for config in model_feature_config.id_list_feature_configs
        ]
        self._id_score_list_feature_names: List[str] = [
            config.name for config in model_feature_config.id_score_list_feature_configs
        ]

        embedding_bag_configs: List[
            EmbeddingBagConfig
        ] = embedding_bag_configs_from_feature_configs(
            [model_feature_config],
        )
        assert (
            embedding_bag_configs
        ), "No embedding bag config generated. Please double check model_feature_config."

        # Assume all id features will be mapped to the same number of dimensions
        assert (
            len({config.embedding_dim for config in embedding_bag_configs}) == 1
        ), "Please ensure all embedding_dims in id_mapping_config are the same"
        embedding_dim = embedding_bag_configs[0].embedding_dim

        self.embedding_bags = torch.nn.ModuleDict(
            {
                table_name: torch.nn.EmbeddingBag(
                    num_embeddings=id_mapping.embedding_table_size,
                    embedding_dim=id_mapping.embedding_dim,
                    mode=str(id_mapping.pooling_type.name).lower(),
                )
                for table_name, id_mapping in model_feature_config.id_mapping_config.items()
            }
        )
        self.feat2table: Dict[str, str] = {
            feature_name: config.id_mapping_name
            for feature_name, config in model_feature_config.name2config.items()
        }
        self._output_dim = (
            state_dense_dim
            + len(self._id_list_feature_names) * embedding_dim
            + len(self._id_score_list_feature_names) * embedding_dim
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def input_prototype(self):
        id_list_features = {
            k: (torch.tensor([0], dtype=torch.long), torch.tensor([], dtype=torch.long))
            for k in self._id_list_feature_names
        }
        id_score_list_features = {
            k: (
                torch.tensor([0], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.float),
            )
            for k in self._id_score_list_feature_names
        }
        return rlt.FeatureData(
            float_features=torch.randn(1, self.state_dense_dim),
            id_list_features_raw=id_list_features,
            id_score_list_features_raw=id_score_list_features,
        )

    def forward(self, state: rlt.FeatureData):
        # id_list is (offset, value); sum pooling
        id_list_embeddings = [
            self.embedding_bags[self.feat2table[feature_name]](input=v[1], offsets=v[0])
            for feature_name, v in state.id_list_features_raw.items()
        ]

        # id_score_list is (offset, key, value); weighted sum pooling
        id_score_list_embeddings = [
            self.embedding_bags[self.feat2table[feature_name]](
                input=v[1], offsets=v[0], per_sample_weights=v[2]
            )
            for feature_name, v in state.id_score_list_features_raw.items()
        ]
        return torch.cat(
            id_list_embeddings + id_score_list_embeddings + [state.float_features],
            dim=1,
        )
