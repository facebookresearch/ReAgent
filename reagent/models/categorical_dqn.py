#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn.functional as F
from ml.rl import types as rlt
from ml.rl.models.base import ModelBase
from ml.rl.models.fully_connected_network import FullyConnectedNetwork


class CategoricalDQN(ModelBase):
    def __init__(
        self,
        state_dim,
        action_dim,
        num_atoms,
        qmin,
        qmax,
        sizes,
        activations,
        use_batch_norm=False,
        dropout_ratio=0.0,
        use_gpu=False,
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
        self.support = torch.linspace(qmin, qmax, num_atoms)
        if use_gpu:
            self.support = self.support.cuda()

    def input_prototype(self):
        return rlt.PreprocessedState.from_tensor(torch.randn(1, self.state_dim))

    def forward(self, input: rlt.PreprocessedState):
        dist = self.log_dist(input).exp()
        q_values = (dist * self.support).sum(2)
        return rlt.AllActionQValues(q_values=q_values)

    def log_dist(self, input: rlt.PreprocessedState):
        log_dist = self.fc(input.state.float_features).reshape(
            -1, self.action_dim, self.num_atoms
        )
        return F.log_softmax(log_dist, -1)

    def serving_model(self):
        return self.cpu_model()
