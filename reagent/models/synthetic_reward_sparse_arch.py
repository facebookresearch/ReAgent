#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from typing import List

import torch
import torch.nn as nn
from reagent.core import types as rlt
from reagent.core.torch_utils import split_sequence_keyed_jagged_tensor
from reagent.models.base import ModelBase
from reagent.models.fully_connected_network import ACTIVATION_MAP
from reagent.models.synthetic_reward import _gen_mask
from torchrec import EmbeddingBagCollection
from torchrec.models.dlrm import SparseArch, InteractionArch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


logger = logging.getLogger(__name__)


def create_dense_arch(
    input_dim: int,
    dense_sizes: List[int],
    dense_activations: List[str],
    use_batch_norm: bool,
    use_layer_norm: bool,
):
    modules: List[nn.Module] = []
    prev_layer_size = input_dim
    for size, activation in zip(dense_sizes, dense_activations):
        if use_batch_norm:
            modules.append(nn.BatchNorm1d(prev_layer_size))
        modules.append(nn.Linear(prev_layer_size, size))
        if use_layer_norm:
            modules.append(nn.LayerNorm(size))
        modules.append(ACTIVATION_MAP[activation]())
        prev_layer_size = size
    return nn.Sequential(*modules)


class SyntheticRewardSparseArchNet(ModelBase):
    """
    This base class provides basic operations to consume inputs and call a synthetic reward net

    A synthetic reward net (self.net) assumes the input contains only torch.Tensors.
    Expected input shape:
        state: seq_len, batch_size, state_dim
        action: seq_len, batch_size, action_dim
    Expected output shape:
        reward: batch_size, seq_len
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, training_batch: rlt.MemoryNetworkInput):
        # state shape: seq_len, batch_size, state_dim
        state = training_batch.state.float_features
        # action shape: seq_len, batch_size, action_dim
        action = training_batch.action.float_features

        # shape: batch_size, 1
        valid_step = training_batch.valid_step
        seq_len, batch_size, _ = training_batch.action.float_features.shape

        # output shape: batch_size, seq_len
        output = self.net(
            state,
            action,
            training_batch.state.id_list_features,
            training_batch.state.id_score_list_features,
            training_batch.action.id_list_features,
            training_batch.action.id_score_list_features,
        )
        assert valid_step is not None
        mask = _gen_mask(valid_step, batch_size, seq_len)
        output_masked = output * mask

        pred_reward = output_masked.sum(dim=1, keepdim=True)
        return rlt.SyntheticRewardNetworkOutput(
            predicted_reward=pred_reward,
            mask=mask,
            output=output,
        )

    def export_mlp(self):
        """
        Export an pytorch nn to feed to predictor wrapper.
        """
        return self.net

    def requires_model_parallel(self):
        return True


class SingleStepSyntheticSparseArchRewardNet(nn.Module):
    def __init__(
        self,
        state_dense_dim: int,
        action_dense_dim: int,
        dense_sizes: List[int],
        dense_activations: List[str],
        overall_sizes: List[int],
        overall_activations: List[str],
        embedding_bag_collection: EmbeddingBagCollection,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
    ):
        """
        Decompose rewards of the last step to all individual steps.

        This model arch accepts sparse features and is similar to / inspired by
        the model in "Deep Learning Recommendation Model for Personalization and
        Recommendation Systems" (https://arxiv.org/abs/1906.00091)

        The model arch can be described as below:


                             last_layer_activation
                                       ^
                                  overall arch
                                       ^
              -----------interaction arch (2D + 2F + F choose 2) -------
              ^                        ^                               ^
        state_dense_out(D)      action_dense_out(D)              sparse_out(F*D)
              ^                        ^                               ^
        state_dense_arch         action_dense_arch                sparse arch
              ^                        ^                               ^
          state_dense             action_dense             state_sparse / action_sparse


        , where:
        D: last layer of dense_sizes (equal to sparse features' embedding_dim)
        F: number of total sparse features (from both state and action and from both
            id-list and id-score-list features)
        Interaction arch returns a concatenation of
            (1) and the dense layers itself,
            (2) the dot product of each sparse embedding with the output of the dense arch,
            (3) the pairwise dot product of each sparse embedding pair,

        """
        super().__init__()
        self.validate_parameters(
            dense_sizes,
            dense_activations,
            overall_sizes,
            overall_activations,
            embedding_bag_collection,
        )

        self.state_dense_arch = create_dense_arch(
            state_dense_dim,
            dense_sizes,
            dense_activations,
            use_batch_norm,
            use_layer_norm,
        )
        self.action_dense_arch = create_dense_arch(
            action_dense_dim,
            dense_sizes,
            dense_activations,
            use_batch_norm,
            use_layer_norm,
        )
        # sparse arch will be shared for state sparse features and action sparse features
        self.sparse_arch = SparseArch(embedding_bag_collection)

        # Overall arch
        F = sum(
            [
                len(conf.feature_names)
                for conf in embedding_bag_collection.embedding_bag_configs
            ]
        )
        D = dense_sizes[-1]
        self.F = F
        self.D = D
        sparse_feature_names = []
        for conf in embedding_bag_collection.embedding_bag_configs:
            sparse_feature_names.extend(conf.feature_names)
        self.inter_arch_sparse_and_state_dense = InteractionArch(
            sparse_feature_names=sparse_feature_names
        )
        self.inter_arch_sparse_and_action_dense = InteractionArch(
            sparse_feature_names=sparse_feature_names
        )

        interaction_output_dim = 2 * D + 2 * F + F * (F - 1) // 2
        self.overall_arch = create_dense_arch(
            interaction_output_dim,
            overall_sizes,
            overall_activations,
            use_batch_norm,
            use_layer_norm,
        )

    def validate_parameters(
        self,
        dense_sizes: List[int],
        dense_activations: List[str],
        overall_sizes: List[int],
        overall_activations: List[str],
        embedding_bag_collection: EmbeddingBagCollection,
    ):
        for i in range(1, len(embedding_bag_collection.embedding_bag_configs)):
            conf_prev = embedding_bag_collection.embedding_bag_configs[i - 1]
            conf = embedding_bag_collection.embedding_bag_configs[i]
            assert (
                conf_prev.embedding_dim == conf.embedding_dim
            ), "All EmbeddingBagConfigs must have the same embedding_dim"

        conf = embedding_bag_collection.embedding_bag_configs[0]
        dense_output_size = dense_sizes[-1]
        assert (
            dense_output_size == conf.embedding_dim
        ), "The last layer of dense_sizes should be equal to embedding_dim of sparse features"
        assert overall_sizes[-1] == 1, "The last layer of overall_sizes should be 1"

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        state_id_list: KeyedJaggedTensor,
        state_id_score_list: KeyedJaggedTensor,
        action_id_list: KeyedJaggedTensor,
        action_id_score_list: KeyedJaggedTensor,
    ):
        # state shape: seq_len, batch_size, state_dim
        # action shape: seq_len, batch_size, action_dim
        # state_sparse: sparse state features from seq_len steps
        seq_len, batch_size, _ = state.shape

        # state_dense_out shape: seq_len, batch_size, embed_dim
        state_dense_out = self.state_dense_arch(state)
        # action_dense_out shape: seq_len, batch_size, embed_dim
        action_dense_out = self.action_dense_arch(action)

        sparse_data_per_step: List[
            KeyedJaggedTensor
        ] = self.create_sparse_data_per_step(
            state_id_list,
            state_id_score_list,
            action_id_list,
            action_id_score_list,
            seq_len,
        )
        sparse_embed_per_step = [
            self.sparse_arch(sparse_data_per_step[i]) for i in range(seq_len)
        ]

        interaction_per_step = []
        for i in range(seq_len):
            # shape: batch_size, D + F + F choose 2
            inter_sparse_state = self.inter_arch_sparse_and_state_dense(
                dense_features=state_dense_out[i],
                sparse_features=sparse_embed_per_step[i],
            )
            # shape: batch_size, D + F + F choose 2
            inter_sparse_action = self.inter_arch_sparse_and_action_dense(
                dense_features=action_dense_out[i],
                sparse_features=sparse_embed_per_step[i],
            )
            # We need to concat interactions of sparse-state and sparse-action
            # However, sparse feature embeddings' self dot-products are included
            # in both interactions so we need to dedup
            # interaction shape: batch_size, 2D + 2F + F choose 2
            interaction = torch.cat(
                (
                    inter_sparse_state,
                    inter_sparse_action[:, : self.D + self.F],
                ),
                dim=1,
            )
            interaction_per_step.append(interaction)

        # interaction_per_step shape: seq_len, batch_size, 2D + 2F + F choose 2
        interaction_per_step = torch.stack(interaction_per_step, dim=0)
        # overall_arch_out shape: seq_len, batch_size, 1
        overall_arch_out = self.overall_arch(interaction_per_step)
        # return shape: batch_size, seq_len
        return overall_arch_out.squeeze(2).transpose(0, 1)

    def create_sparse_data_per_step(
        self,
        state_id_list: KeyedJaggedTensor,
        state_id_score_list: KeyedJaggedTensor,
        action_id_list: KeyedJaggedTensor,
        action_id_score_list: KeyedJaggedTensor,
        seq_len: int,
    ):
        """
        Return a list of KeyedJaggedTensor, where each KeyedJaggedTensor
        represents one step's sparse data.

        Under the hood, we perform the following steps:
        1. Split state_id_list, state_id_score_list, action_id_list, and
        action_id_score_list by steps
        2. Treat state_id_list and action_id_list features as id_score_list
        features with weight=1
        3. Concatenate state_id_list, state_id_score_list, action_id_list, and
        action_id_score_list at each step
        """
        # Convert id_list data as id score list data with weight = 1
        state_id_list._weights = torch.ones_like(state_id_list.values())
        action_id_list._weights = torch.ones_like(action_id_list.values())

        # For each step, we merge all sparse data into one KeyedJaggedTensor
        state_id_list_per_step = split_sequence_keyed_jagged_tensor(
            state_id_list, seq_len
        )
        state_id_score_list_per_step = split_sequence_keyed_jagged_tensor(
            state_id_score_list, seq_len
        )
        action_id_list_per_step = split_sequence_keyed_jagged_tensor(
            action_id_list, seq_len
        )
        action_id_score_list_per_step = split_sequence_keyed_jagged_tensor(
            action_id_score_list, seq_len
        )
        sparse_data_per_step = [
            KeyedJaggedTensor.concat(
                KeyedJaggedTensor.concat(
                    state_id_list_per_step[i], action_id_list_per_step[i]
                ),
                KeyedJaggedTensor.concat(
                    state_id_score_list_per_step[i], action_id_score_list_per_step[i]
                ),
            )
            for i in range(seq_len)
        ]
        return sparse_data_per_step
