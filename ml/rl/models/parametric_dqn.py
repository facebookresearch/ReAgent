#!/usr/bin/env python3

import torch
from ml.rl import types as rlt
from ml.rl.models.base import ModelBase
from ml.rl.models.fully_connected_network import FullyConnectedNetwork


class FullyConnectedParametricDQN(ModelBase):
    def __init__(self, state_dim, action_dim, sizes, activations, use_batch_norm=False):
        super(FullyConnectedParametricDQN, self).__init__()
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
            [state_dim + action_dim] + sizes + [1],
            activations + ["linear"],
            use_batch_norm=use_batch_norm,
        )

    def input_prototype(self):
        return rlt.StateAction(
            state=rlt.FeatureVector(float_features=torch.randn(1, self.state_dim)),
            action=rlt.FeatureVector(float_features=torch.randn(1, self.action_dim)),
        )

    def forward(self, input):
        cat_input = torch.cat(
            (input.state.float_features, input.action.float_features), dim=1
        )
        q_value = self.fc(cat_input)
        return rlt.SingleQValue(q_value=q_value)
