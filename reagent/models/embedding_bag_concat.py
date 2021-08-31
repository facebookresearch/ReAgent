#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Dict, List

import torch
from reagent.core import types as rlt
from reagent.models.base import ModelBase


class EmbeddingBagConcat(ModelBase):
    """
    Concatenating embedding with float features before passing the input
    to DQN
    """

    def __init__(
        self,
        state_dim: int,
        model_feature_config: rlt.ModelFeatureConfig,
        embedding_dim: int,
    ):
        super().__init__()
        assert state_dim > 0, "state_dim must be > 0, got {}".format(state_dim)
        self.state_dim = state_dim
        # for input prototype
        self._id_list_feature_names: List[str] = [
            config.name for config in model_feature_config.id_list_feature_configs
        ]
        self._id_score_list_feature_names: List[str] = [
            config.name for config in model_feature_config.id_score_list_feature_configs
        ]

        self.embedding_bags = torch.nn.ModuleDict(
            {
                table_name: torch.nn.EmbeddingBag(
                    num_embeddings=id_mapping.value.table_size,
                    embedding_dim=embedding_dim,
                    mode="sum",
                )
                for table_name, id_mapping in model_feature_config.id_mapping_config.items()
            }
        )
        self.feat2table: Dict[str, str] = {
            feature_name: config.id_mapping_name
            for feature_name, config in model_feature_config.name2config.items()
        }
        self._output_dim = (
            state_dim
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
            float_features=torch.randn(1, self.state_dim),
            id_list_features=id_list_features,
            id_score_list_features=id_score_list_features,
        )

    def forward(self, state: rlt.FeatureData):
        # id_list is (offset, value); sum pooling
        id_list_embeddings = [
            self.embedding_bags[self.feat2table[feature_name]](input=v[1], offsets=v[0])
            for feature_name, v in state.id_list_features.items()
        ]

        # id_score_list is (offset, key, value); weighted sum pooling
        id_score_list_embeddings = [
            self.embedding_bags[self.feat2table[feature_name]](
                input=v[1], offsets=v[0], per_sample_weights=v[2]
            )
            for feature_name, v in state.id_score_list_features.items()
        ]
        return torch.cat(
            id_list_embeddings + id_score_list_embeddings + [state.float_features],
            dim=1,
        )
