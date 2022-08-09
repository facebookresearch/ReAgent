#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import List

import torch
from reagent.core import types as rlt
from reagent.models import FullyConnectedNetwork
from reagent.models.base import ModelBase
from torchrec.models.dlrm import SparseArch
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class SparseDQN(ModelBase):
    """
    Concatenating embeddings from bag collection with float features before passing the input
    to DQN
    """

    def __init__(
        self,
        state_dense_dim: int,
        embedding_bag_collection: EmbeddingBagCollection,
        action_dense_dim: int,
        overarch_dims: List[int],
        activation: str = "relu",
        final_activation: str = "relu",
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.sparse_arch: SparseArch = SparseArch(embedding_bag_collection)

        self.sparse_embedding_dim: int = sum(
            [
                len(embc.feature_names) * embc.embedding_dim
                for embc in embedding_bag_collection.embedding_bag_configs()
            ]
        )

        self.input_dim: int = (
            state_dense_dim + self.sparse_embedding_dim + action_dense_dim
        )
        layers = [self.input_dim] + overarch_dims + [output_dim]
        activations = [activation] * len(overarch_dims) + [final_activation]
        self.q_network = FullyConnectedNetwork(
            layers=layers,
            activations=activations,
            use_batch_norm=use_batch_norm,
        )

    def fetch_id_list_features(
        self, state: rlt.FeatureData, action: rlt.FeatureData
    ) -> KeyedJaggedTensor:
        assert state.id_list_features is not None or action.id_list_features is not None
        if state.id_list_features is not None and action.id_list_features is None:
            sparse_features = state.id_list_features
        elif state.id_list_features is None and action.id_list_features is not None:
            sparse_features = action.id_list_features
        elif state.id_list_features is not None and action.id_list_features is not None:
            sparse_features = KeyedJaggedTensor.concat(
                [state.id_list_features, action.id_list_features]
            )
        else:
            raise ValueError
        # TODO: add id_list_score_features
        return sparse_features

    def forward(self, state: rlt.FeatureData, action: rlt.FeatureData) -> torch.Tensor:
        dense_features = torch.cat(
            (state.float_features, action.float_features), dim=-1
        )
        batch_size = dense_features.shape[0]
        sparse_features = self.fetch_id_list_features(state, action)
        # shape: batch_size, num_sparse_features, embedding_dim
        embedded_sparse = self.sparse_arch(sparse_features)
        embedded_sparse = embedded_sparse.reshape(batch_size, -1)
        concatenated_dense = torch.cat((dense_features, embedded_sparse), dim=-1)

        return self.q_network(concatenated_dense)
