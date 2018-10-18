#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from ml.rl import types as rlt
from ml.rl.models.base import ModelBase
from ml.rl.models.fully_connected_network import FullyConnectedNetwork


class FullyConnectedActor(ModelBase):
    def __init__(
        self,
        state_dim,
        action_dim,
        sizes,
        activations,
        use_batch_norm=False,
        action_activation="tanh",
    ):
        super(FullyConnectedActor, self).__init__()
        assert state_dim > 0, "state_dim must be > 0, got {}".format(state_dim)
        assert action_dim > 0, "action_dim must be > 0, got {}".format(action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        assert len(sizes) == len(
            activations
        ), "The numbers of sizes and activations must match; got {} vs {}".format(
            len(sizes), len(activations)
        )
        self.action_activation = action_activation
        self.fc = FullyConnectedNetwork(
            [state_dim] + sizes + [action_dim],
            activations + [self.action_activation],
            use_batch_norm=use_batch_norm,
        )

    def input_prototype(self):
        return rlt.State(float_features=torch.randn(1, self.state_dim))

    def forward(self, input):
        action = self.fc(input.float_features)
        return rlt.ParametricAction(float_features=action)
