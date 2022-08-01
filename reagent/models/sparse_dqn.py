#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import List

import torch
from reagent.core import types as rlt
from reagent.models import FullyConnectedNetwork
from reagent.models.base import ModelBase
from torchrec.models.dlrm import SparseArch
from torchrec.modules.embedding_modules import EmbeddingBagCollection


class SparseDQN(ModelBase):
    """
    Concatenating embeddings from bag collection with float features before passing the input
    to DQN
    """

    def __init__(
        self,
        state_dense_dim: int,
        embedding_bag_collection_for_features: EmbeddingBagCollection,
        action_dense_dim: int,
        embedding_bag_collection_for_actions: EmbeddingBagCollection,
        overarch_dims: List[int],
        activation: str = "relu",
        final_activation: str = "relu",
        use_batch_norm="True",
        use_layer_norm="False",
        output_dim=1,
    ) -> None:
        super().__init__()
        self.sparse_arch_for_features: SparseArch = SparseArch(
            embedding_bag_collection_for_features
        )

        self.sparse_arch_for_actions: SparseArch = SparseArch(
            embedding_bag_collection_for_actions
        )

        self.sparse_embedding_dim: int = sum(
            [
                len(embc.feature_names) * embc.embedding_dim
                for embc in embedding_bag_collection_for_features.embedding_bag_configs()
            ]
        )


        self.sparse_action_dim: int = sum(
            [
                len(embc.feature_names) * embc.embedding_dim
                for embc in embedding_bag_collection_for_actions.embedding_bag_configs()
            ]
        )

        self.input_dim: int = (
            state_dense_dim + self.sparse_embedding_dim +  self.sparse_action_dim
        )
        layers = [self.input_dim] + overarch_dims + [output_dim]
        activations = [activation] * len(overarch_dims) + [final_activation]
        self.q_network = FullyConnectedNetwork(
            layers=layers,
            activations=activations,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, state: rlt.FeatureData, action: rlt.FeatureData) -> torch.Tensor:
        dense_features = torch.cat(
            (state.float_features, action.float_features), dim=-1
        )
        sparse_features = state.id_list_features
        embedded_sparse = self.sparse_arch_for_features(sparse_features)
        concatenated_dense = torch.cat((dense_features, embedded_sparse), dim=-1)

        return self.q_network(concatenated_dense)

    def export_mlp(self):
        return self.q_network.get_target_network().dnn
