#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional

import torch
from reagent import types as rlt
from reagent.models.base import ModelBase
from reagent.models.fully_connected_network import FullyConnectedNetwork


class FullyConnectedDQN(ModelBase):
    def __init__(
        self,
        state_dim,
        action_dim,
        sizes,
        activations,
        *,
        num_atoms: Optional[int] = None,
        use_batch_norm=False,
        dropout_ratio=0.0,
        normalized_output=False,
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
        self.num_atoms = num_atoms
        self.fc = FullyConnectedNetwork(
            [state_dim] + sizes + [action_dim * (num_atoms or 1)],
            activations + ["linear"],
            use_batch_norm=use_batch_norm,
            dropout_ratio=dropout_ratio,
            normalize_output=normalized_output,
        )

    def input_prototype(self):
        return rlt.FeatureData(self.fc.input_prototype())

    def forward(self, state: rlt.FeatureData) -> torch.Tensor:
        float_features = state.float_features
        x = self.fc(float_features)
        if self.num_atoms is not None:
            x = x.view(float_features.shape[0], self.action_dim, self.num_atoms)
        return x
