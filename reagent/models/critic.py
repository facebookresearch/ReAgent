#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import List

import torch
from reagent import types as rlt
from reagent.models.base import ModelBase
from reagent.models.fully_connected_network import FullyConnectedNetwork


class FullyConnectedCritic(ModelBase):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sizes: List[int],
        activations: List[str],
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        output_dim: int = 1,
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
        self.fc = FullyConnectedNetwork(
            [state_dim + action_dim] + sizes + [output_dim],
            activations + ["linear"],
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
        assert (
            len(state.float_features.shape) == len(action.float_features.shape)
            and len(action.float_features.shape) == 2
            and (state.float_features.shape[0] == action.float_features.shape[0])
        ), (
            f"state shape: {state.float_features.shape}; action shape: "
            f"{action.float_features.shape} not equal to (batch_size, feature_dim)"
        )
        cat_input = torch.cat((state.float_features, action.float_features), dim=-1)
        return self.fc(cat_input)
