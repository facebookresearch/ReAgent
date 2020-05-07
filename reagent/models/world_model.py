#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from reagent import types as rlt
from reagent.models.base import ModelBase
from reagent.models.mdn_rnn import MDNRNN


class MemoryNetwork(ModelBase):
    def __init__(
        self, state_dim, action_dim, num_hiddens, num_hidden_layers, num_gaussians
    ):
        super().__init__()
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

    def input_prototype(self):
        return (
            rlt.FeatureData(torch.randn(1, 1, self.state_dim)),
            rlt.FeatureData(torch.randn(1, 1, self.action_dim)),
        )

    def forward(self, state: rlt.FeatureData, action: rlt.FeatureData):
        (
            mus,
            sigmas,
            logpi,
            rewards,
            not_terminals,
            all_steps_hidden,
            last_step_hidden_and_cell,
        ) = self.mdnrnn(action.float_features, state.float_features)
        return rlt.MemoryNetworkOutput(
            mus=mus,
            sigmas=sigmas,
            logpi=logpi,
            reward=rewards,
            not_terminal=not_terminals,
            last_step_lstm_hidden=last_step_hidden_and_cell[0],
            last_step_lstm_cell=last_step_hidden_and_cell[1],
            all_steps_lstm_hidden=all_steps_hidden,
        )
