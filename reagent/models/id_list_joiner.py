#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from reagent import types as rlt
from reagent.models.base import ModelBase


class IdListJoiner(ModelBase):
    """
    Concatenating embedding with float features
    """

    def __init__(
        self,
        model_feature_config: rlt.ModelFeatureConfig,
        input_dim: int,
        embedding_dim: int,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = (
            input_dim
            + len(model_feature_config.id_list_feature_configs) * embedding_dim
        )
        self.embedding_bags = torch.nn.ModuleDict(  # type: ignore
            {
                id_list_feature.name: torch.nn.EmbeddingBag(  # type: ignore
                    len(
                        model_feature_config.id_mapping_config[
                            id_list_feature.id_mapping_name
                        ].ids
                    ),
                    embedding_dim,
                )
                for id_list_feature in model_feature_config.id_list_feature_configs
            }
        )

    def input_dim(self):
        return self._input_dim

    def output_dim(self):
        return self._output_dim

    def input_prototype(self):
        return rlt.PreprocessedState(
            state=rlt.PreprocessedFeatureVector(
                float_features=torch.randn(1, self._input_dim),
                id_list_features={
                    k: (
                        torch.zeros(1, dtype=torch.long),
                        torch.ones(1, dtype=torch.long),
                    )
                    for k in self.embedding_bags
                },
            )
        )

    def forward(self, input: rlt.PreprocessedState) -> torch.Tensor:
        embeddings = [
            m(
                input.state.id_list_features[name][1],
                input.state.id_list_features[name][0],
            )
            for name, m in self.embedding_bags.items()
        ]
        return torch.cat(embeddings + [input.state.float_features], dim=1)
