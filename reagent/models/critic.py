#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import List

import torch
import torch.fx
from reagent.core import types as rlt
from reagent.models.base import ModelBase
from reagent.models.fully_connected_network import FullyConnectedNetwork


# This method contains dynamic control flow
# Use torch.fx.wrap to mark it as a leaf module for FX tracing
@torch.fx.wrap
def run_feature_validation(
    state_float_features_dim: int,
    action_float_features_dim: int,
    state_float_features_batch_size: int,
    action_float_features_batch_size: int,
) -> None:

    assert (
        state_float_features_dim == 2
    ), f"Expected dimension of state is 2. Got {state_float_features_dim}"

    assert (
        action_float_features_dim == state_float_features_dim
    ), "Dimensions of state and action mismatch"

    assert (
        state_float_features_batch_size == action_float_features_batch_size
    ), "Batch sizes of state and action mismatch"


class FullyConnectedCritic(ModelBase):
    """
    A general model arch for mapping from state and action to scalar values.

    The model arch is often used to implement the critic in actor-critic algorithms.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sizes: List[int],
        activations: List[str],
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        output_dim: int = 1,
        final_activation: str = "linear",  # most of the time "linear" is the right final activation to use!
    ) -> None:
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
        self.fc = FullyConnectedNetwork(
            [state_dim + action_dim] + sizes + [output_dim],
            activations + [final_activation],
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

    def input_prototype(self):
        # for inference: (batchsize, feature_dim)
        return (
            rlt.FeatureData(torch.randn(1, self.state_dim)),
            rlt.FeatureData(torch.randn(1, self.action_dim)),
        )

    def forward(self, state: rlt.FeatureData, action: rlt.FeatureData):
        state_float_features_dim = state.float_features.dim()
        action_float_features_dim = action.float_features.dim()
        state_float_features_batch_size = state.float_features.size(dim=0)
        action_float_features_batch_size = action.float_features.size(dim=0)

        run_feature_validation(
            state_float_features_dim,
            action_float_features_dim,
            state_float_features_batch_size,
            action_float_features_batch_size,
        )
        cat_input = torch.cat((state.float_features, action.float_features), dim=-1)
        return self.fc(cat_input)
