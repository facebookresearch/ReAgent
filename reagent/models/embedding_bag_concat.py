#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from reagent import types as rlt
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

        self.id_list_embeddings_bag = torch.nn.ModuleDict(
            {
                id_list_feature.name: torch.nn.EmbeddingBag(
                    num_embeddings=len(
                        model_feature_config.id_mapping_config[
                            id_list_feature.id_mapping_name
                        ].ids
                    ),
                    embedding_dim=embedding_dim,
                )
                for id_list_feature in model_feature_config.id_list_feature_configs
            }
        )

        self.id_score_list_embeddings_bag = torch.nn.ModuleDict(
            {
                id_score_list_feature.name: torch.nn.EmbeddingBag(
                    num_embeddings=len(
                        model_feature_config.id_mapping_config[
                            id_score_list_feature.id_mapping_name
                        ].ids
                    ),
                    embedding_dim=embedding_dim,
                    mode="sum",
                )
                for id_score_list_feature in model_feature_config.id_score_list_feature_configs
            }
        )

        self._output_dim = (
            state_dim
            + len(model_feature_config.id_list_feature_configs) * embedding_dim
            + len(model_feature_config.id_score_list_feature_configs) * embedding_dim
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def input_prototype(self):
        return rlt.FeatureData(
            float_features=torch.randn(1, self.state_dim),
            id_list_features={
                k: (torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.long))
                for k in self.id_list_embeddings_bag
            },
            id_score_list_features={
                k: (
                    torch.zeros(1, dtype=torch.long),
                    torch.ones(1, dtype=torch.long),
                    torch.ones(1, dtype=torch.float),
                )
                for k in self.id_score_list_embeddings_bag
            },
        )

    def forward(self, state: rlt.FeatureData):
        # id_list is (offset, value)
        id_list_embeddings = [
            m(input=state.id_list_features[k][1], offsets=state.id_list_features[k][0])
            for k, m in self.id_list_embeddings_bag.items()
        ]

        # id_score_list is (offset, key, value)
        # FIXME: normalize weights (values) so that each "row" sums to 1?
        id_score_list_embeddings = [
            m(
                input=state.id_score_list_features[k][1],
                offsets=state.id_score_list_features[k][0],
                per_sample_weights=state.id_score_list_features[k][2],
            )
            for k, m in self.id_score_list_embeddings_bag.items()
        ]
        return torch.cat(
            id_list_embeddings + id_score_list_embeddings + [state.float_features],
            dim=1,
        )
