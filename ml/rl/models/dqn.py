#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from ml.rl import types as rlt
from ml.rl.models.base import ModelBase
from ml.rl.models.fully_connected_network import FullyConnectedNetwork
from torch.nn.parallel.distributed import DistributedDataParallel


class FullyConnectedDQN(ModelBase):
    def __init__(
        self,
        state_dim,
        action_dim,
        sizes,
        activations,
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
            [state_dim] + sizes + [action_dim],
            activations + ["linear"],
            use_batch_norm=use_batch_norm,
            dropout_ratio=dropout_ratio,
        )

    def get_distributed_data_parallel_model(self):
        return _DistributedDataParallelFullyConnectedDQN(self)

    def input_prototype(self):
        return rlt.StateInput(
            state=rlt.FeatureVector(float_features=torch.randn(1, self.state_dim))
        )

    def forward(self, input):
        q_values = self.fc(input.state.float_features)
        return rlt.AllActionQValues(q_values=q_values)


class _DistributedDataParallelFullyConnectedDQN(ModelBase):
    def __init__(self, fc_dqn):
        super().__init__()
        self.state_dim = fc_dqn.state_dim
        self.action_dim = fc_dqn.action_dim
        current_device = torch.cuda.current_device()
        self.data_parallel = DistributedDataParallel(
            fc_dqn.fc, device_ids=[current_device], output_device=current_device
        )
        self.fc_dqn = fc_dqn

    def input_prototype(self):
        return rlt.StateInput(
            state=rlt.FeatureVector(float_features=torch.randn(1, self.state_dim))
        )

    def cpu_model(self):
        return self.fc_dqn.cpu_model()

    def forward(self, input):
        q_values = self.data_parallel(input.state.float_features)
        return rlt.AllActionQValues(q_values=q_values)
