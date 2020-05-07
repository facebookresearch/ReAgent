#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from reagent import types as rlt
from reagent.models.base import ModelBase
from reagent.models.fully_connected_network import FullyConnectedNetwork


class FullyConnectedCritic(ModelBase):
    def __init__(
        self,
        state_dim,
        action_dim,
        sizes,
        activations,
        use_batch_norm=False,
        use_layer_norm=False,
        output_dim=1,
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
        return rlt.PreprocessedStateAction.from_tensors(
            state=torch.randn(1, self.state_dim), action=torch.randn(1, self.action_dim)
        )

    def forward(self, input: rlt.PreprocessedStateAction):
        cat_input = torch.cat(
            (input.state.float_features, input.action.float_features), dim=1
        )
        return self.fc(cat_input)
