#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from reagent import types as rlt
from reagent.models.base import ModelBase
from reagent.models.fully_connected_network import FullyConnectedNetwork


class QuantileDQN(ModelBase):
    def __init__(
        self,
        state_dim,
        action_dim,
        sizes,
        activations,
        num_atoms=50,
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
        self.fc = FullyConnectedNetwork(
            [state_dim] + sizes + [action_dim * num_atoms],
            activations + ["linear"],
            use_batch_norm=use_batch_norm,
            dropout_ratio=dropout_ratio,
        )

        self.num_atoms = num_atoms
        self.action_dim = action_dim

    def input_prototype(self):
        return rlt.PreprocessedState.from_tensor(torch.randn(1, self.state_dim))

    def forward(self, input: rlt.PreprocessedState):
        q_values = self.dist(input).mean(dim=2)
        return rlt.AllActionQValues(q_values=q_values)

    def dist(self, input: rlt.PreprocessedState):
        return self.fc(input.state.float_features).reshape(
            -1, self.action_dim, self.num_atoms
        )
