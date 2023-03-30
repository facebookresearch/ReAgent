import unittest

import torch
from reagent.models.sparse_dqn import SparseDQN
from torchrec import EmbeddingBagCollection, EmbeddingBagConfig


class TestSparseDQN(unittest.TestCase):
    def test_single_step_sparse_dqn(self):
        state_dense_dim = 10
        action_dense_dim = 2
        dense_sizes = [256, 32]
        activation = "relu"
        final_activation = "relu"
        # Fake embedding bag configs
        embedding_table_size = 1000
        embedding_dim = 32
        num_sparse_features = 2  # refer to watched_ids and liked_ids below
        embedding_bag_configs = [
            EmbeddingBagConfig(
                name="video_id",
                feature_names=["watched_ids", "liked_ids"],
                num_embeddings=embedding_table_size,
                embedding_dim=embedding_dim,
            )
        ]

        num_sparse_features_ro = 2  # refer to watched_page_ids and liked_ids below
        embedding_bag_configs_ro = [
            EmbeddingBagConfig(
                name="watched_page_ids",
                feature_names=["watched_page_ids", "liked_ids"],
                num_embeddings=embedding_table_size,
                embedding_dim=embedding_dim,
            )
        ]
        embedding_bag_col = EmbeddingBagCollection(
            device=torch.device("cpu"), tables=embedding_bag_configs
        )
        embedding_bag_col_ro = EmbeddingBagCollection(
            device=torch.device("cpu"), tables=embedding_bag_configs_ro
        )

        net = SparseDQN(
            state_dense_dim=state_dense_dim,
            embedding_bag_collection=embedding_bag_col,
            embedding_bag_collection_ro=embedding_bag_col_ro,
            action_dense_dim=action_dense_dim,
            overarch_dims=dense_sizes,
            activation=activation,
            final_activation=final_activation,
            output_dim=action_dense_dim,
        ).q_network.dnn

        # the dim of the input to overall arch is dimension of dense features plus
        # number of sparse features times embedding dimension for sparse features
        assert (
            net[1].in_features
            == state_dense_dim
            + action_dense_dim
            + num_sparse_features * embedding_dim
            + num_sparse_features_ro * embedding_dim
        )
        assert net[1].out_features == dense_sizes[0]
        assert net[4].in_features == dense_sizes[0]
        assert net[4].out_features == dense_sizes[1]
        assert net[7].in_features == dense_sizes[1]

        assert net[7].out_features == action_dense_dim
