#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from ml.rl import types as rlt
from ml.rl.models.base import ModelBase


class BatchConstrainedDQN(ModelBase):
    def __init__(self, state_dim, q_network, imitator_network, bcq_drop_threshold):
        super(BatchConstrainedDQN, self).__init__()
        assert state_dim > 0, "state_dim must be > 0, got {}".format(state_dim)
        self.state_dim = state_dim
        self.q_network = q_network
        self.imitator_network = imitator_network
        self.invalid_action_penalty = -1e10
        self.bcq_drop_threshold = bcq_drop_threshold

    def input_prototype(self):
        return rlt.StateInput(
            state=rlt.FeatureVector(float_features=torch.randn(1, self.state_dim))
        )

    def forward(self, input):
        q_values = self.q_network(input)
        imitator_probs = self.imitator_network(input.state.float_features)
        invalid_actions = (imitator_probs < self.bcq_drop_threshold).float()
        invalid_action_penalty = self.invalid_action_penalty * invalid_actions
        constrained_q_values = q_values.q_values + invalid_action_penalty
        return rlt.AllActionQValues(q_values=constrained_q_values)
