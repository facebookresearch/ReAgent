#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from reagent import types as rlt
from reagent.models.base import ModelBase
from reagent.models.fully_connected_network import FullyConnectedNetwork


class FullyConnectedDQNWithEmbedding(ModelBase):
    """
    Concatenating embedding with float features before passing the input
    to DQN
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        sizes,
        activations,
        model_feature_config: rlt.ModelFeatureConfig,
        embedding_dim: int,
        use_batch_norm=False,
        dropout_ratio=0.0,
    ):
        super().__init__()
        assert state_dim > 0, "state_dim must be > 0, got {}".format(state_dim)
        assert action_dim > 0, "action_dim must be > 0, got {}".format(action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        assert len(sizes) == len(
            activations
        ), "The numbers of sizes and activations must match; got {} vs {}".format(
            len(sizes), len(activations)
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

        fc_input_dim = (
            state_dim
            + len(model_feature_config.id_list_feature_configs) * embedding_dim
        )

        self.fc = FullyConnectedNetwork(
            [fc_input_dim] + sizes + [action_dim],
            activations + ["linear"],
            use_batch_norm=use_batch_norm,
            dropout_ratio=dropout_ratio,
        )

    def input_prototype(self):
        return rlt.PreprocessedState(
            state=rlt.PreprocessedFeatureVector(
                float_features=torch.randn(1, self.state_dim),
                id_list_features={
                    k: (
                        torch.zeros(1, dtype=torch.long),
                        torch.ones(1, dtype=torch.long),
                    )
                    for k in self.embedding_bags
                },
            )
        )

    def forward(self, input: rlt.PreprocessedState):
        embeddings = [
            m(
                input.state.id_list_features[name][1],
                input.state.id_list_features[name][0],
            )
            for name, m in self.embedding_bags.items()
        ]
        fc_input = torch.cat(embeddings + [input.state.float_features], dim=1)
        q_values = self.fc(fc_input)
        return rlt.AllActionQValues(q_values=q_values)
