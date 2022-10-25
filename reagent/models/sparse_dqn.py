#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import List, Optional, Tuple

import torch
from reagent.core import types as rlt
from reagent.models import FullyConnectedNetwork
from reagent.models.base import ModelBase
from torchrec.models.dlrm import SparseArch
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


# troch.fx.trace does not support dynamic control flow, wrap the if-else and assert logic in this function to work around this limitation
@torch.fx.wrap
def fetch_id_list_features(
    state: rlt.FeatureData, action: rlt.FeatureData
) -> Tuple[Optional[KeyedJaggedTensor], Optional[KeyedJaggedTensor]]:
    assert (
        state.id_list_features is not None
        or state.id_list_features_ro is not None
        or action.id_list_features is not None
        or action.id_list_features_ro is not None
    )

    def _get_sparse_features(
        id_list_features_1, id_list_features_2
    ) -> Optional[KeyedJaggedTensor]:
        sparse_features = None
        if id_list_features_1 is not None and id_list_features_2 is None:
            sparse_features = id_list_features_1
        elif id_list_features_1 is None and id_list_features_2 is not None:
            sparse_features = id_list_features_2
        elif id_list_features_1 is not None and id_list_features_2 is not None:
            sparse_features = KeyedJaggedTensor.concat(
                [id_list_features_1, id_list_features_2]
            )
        return sparse_features

    sparse_features = _get_sparse_features(
        state.id_list_features, action.id_list_features
    )
    sparse_features_ro = _get_sparse_features(
        state.id_list_features_ro, action.id_list_features_ro
    )
    if sparse_features is None and sparse_features_ro is None:
        raise ValueError

    # TODO: add id_list_score_features
    return sparse_features, sparse_features_ro


class SparseDQN(ModelBase):
    """
    Concatenating embeddings from bag collection with float features before passing the input
    to DQN
    """

    def __init__(
        self,
        state_dense_dim: int,
        embedding_bag_collection: Optional[EmbeddingBagCollection],
        embedding_bag_collection_ro: Optional[EmbeddingBagCollection],
        action_dense_dim: int,
        overarch_dims: List[int],
        activation: str = "relu",
        final_activation: str = "relu",
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.sparse_arch: Optional[SparseArch] = (
            SparseArch(embedding_bag_collection) if embedding_bag_collection else None
        )

        self.sparse_arch_ro: Optional[SparseArch] = (
            SparseArch(embedding_bag_collection_ro)
            if embedding_bag_collection_ro
            else None
        )
        self.sparse_embedding_dim: int = (
            sum(
                [
                    len(embc.feature_names) * embc.embedding_dim
                    for embc in embedding_bag_collection.embedding_bag_configs()
                ]
            )
            if embedding_bag_collection is not None
            else 0
        )

        self.sparse_embedding_dim_ro: int = (
            sum(
                [
                    len(embc.feature_names) * embc.embedding_dim
                    for embc in embedding_bag_collection_ro.embedding_bag_configs()
                ]
            )
            if embedding_bag_collection_ro is not None
            else 0
        )

        self.input_dim: int = (
            state_dense_dim
            + self.sparse_embedding_dim
            + self.sparse_embedding_dim_ro
            + action_dense_dim
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
        batch_size = dense_features.shape[0]
        sparse_features, sparse_features_ro = fetch_id_list_features(state, action)
        # shape: batch_size, num_sparse_features, embedding_dim
        embedded_sparse = (
            self.sparse_arch(sparse_features) if self.sparse_arch else None
        )
        embedded_sparse_ro = (
            self.sparse_arch_ro(sparse_features_ro) if self.sparse_arch_ro else None
        )
        features_list: List[torch.Tensor] = [dense_features]
        if embedded_sparse is not None:
            # shape: batch_size, num_sparse_features * embedding_dim
            embedded_sparse = embedded_sparse.reshape(batch_size, -1)
            features_list.append(embedded_sparse)
        if embedded_sparse_ro is not None:
            # shape: batch_size, num_sparse_features * embedding_dim
            embedded_sparse_ro = embedded_sparse_ro.reshape(batch_size, -1)
            features_list.append(embedded_sparse_ro)

        concatenated_dense = torch.cat(features_list, dim=-1)
        return self.q_network(concatenated_dense)
