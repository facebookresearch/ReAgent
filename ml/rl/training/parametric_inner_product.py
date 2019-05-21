#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn


class ParametricInnerProduct(nn.Module):
    def __init__(self, state_model, action_model, state_dim, action_dim):
        super().__init__()
        self.state_model = state_model
        self.action_model = action_model
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, input):
        states = input[:, : self.state_dim]
        actions = input[:, self.state_dim :]
        state_embeddings = self.state_model(states)
        action_embeddings = self.action_model(actions)
        return torch.bmm(
            state_embeddings.view(state_embeddings.size()[0], 1, -1),
            action_embeddings.view(action_embeddings.size()[0], -1, 1),
        ).squeeze(dim=2)
