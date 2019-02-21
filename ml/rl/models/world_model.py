#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from ml.rl import types as rlt
from ml.rl.models.base import ModelBase
from ml.rl.models.mdn_rnn import MDNRNN


class MemoryNetwork(ModelBase):
    def __init__(
        self, state_dim, action_dim, num_hiddens, num_hidden_layers, num_gaussians
    ):
        super(MemoryNetwork, self).__init__()
        self.mdnrnn = MDNRNN(
            state_dim=state_dim,
            action_dim=action_dim,
            num_hiddens=num_hiddens,
            num_hidden_layers=num_hidden_layers,
            num_gaussians=num_gaussians,
        )
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_hiddens = num_hiddens
        self.num_hidden_layers = num_hidden_layers
        self.num_gaussians = num_gaussians

    def get_data_parallel_model(self):
        return _DataParallelMemoryNetwork(self)

    def input_prototype(self):
        return rlt.StateAction(
            state=rlt.FeatureVector(float_features=torch.randn(1, 1, self.state_dim)),
            action=rlt.FeatureVector(float_features=torch.randn(1, 1, self.action_dim)),
        )

    def forward(self, input):
        mus, sigmas, logpi, rewards, not_terminals, next_hiddens = self.mdnrnn(
            input.action.float_features, input.state.float_features
        )
        return rlt.MemoryNetworkOutput(
            mus=mus,
            sigmas=sigmas,
            logpi=logpi,
            reward=rewards,
            not_terminal=not_terminals,
            next_lstm_hidden=next_hiddens[0],
            next_lstm_cell=next_hiddens[1],
        )


class _DataParallelMemoryNetwork(ModelBase):
    def __init__(self, mem_net):
        super(_DataParallelMemoryNetwork, self).__init__()
        self.num_hiddens = mem_net.num_hiddens
        self.num_hidden_layers = mem_net.num_hidden_layers
        self.state_dim = mem_net.state_dim
        self.action_dim = mem_net.action_dim
        self.num_gaussians = mem_net.num_gaussians

        self.data_parallel = torch.nn.DataParallel(mem_net.mdnrnn)
        self.mem_net = mem_net

    def input_prototype(self):
        return rlt.StateAction(
            state=rlt.FeatureVector(float_features=torch.randn(1, 1, self.state_dim)),
            action=rlt.FeatureVector(float_features=torch.randn(1, 1, self.action_dim)),
        )

    def cpu_model(self):
        return self.mem_net.cpu_model()

    def forward(self, input):
        mus, sigmas, logpi, rewards, not_terminals, next_hiddens = self.data_parallel(
            input.action.float_features, input.state.float_features
        )
        return rlt.MemoryNetworkOutput(
            mus=mus,
            sigmas=sigmas,
            logpi=logpi,
            reward=rewards,
            terminal=not_terminals,
            next_lstm_hidden=next_hiddens[0],
            next_lstm_cell=next_hiddens[1],
        )
