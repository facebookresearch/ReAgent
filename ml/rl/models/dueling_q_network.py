#!/usr/bin/env python3

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ml.rl.models.fully_connected_network import gaussian_fill_w_gain


logger = logging.getLogger(__name__)


class DuelingQNetwork(nn.Module):
    def __init__(self, layers, activations, use_batch_norm=False, action_dim=0) -> None:
        """
        Dueling Q-Network Architecture: https://arxiv.org/abs/1511.06581

        :param layers: List of layer dimensions
        :param activations: List of layer activations
        :param use_batch_norm: bool indicating whether to apply batch normalization
        :param action_dim: if !=0 use parametric dueling DQN, else standard dueling DQN
        """
        super(DuelingQNetwork, self).__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        self.batch_norm_ops: nn.ModuleList = nn.ModuleList()
        self.activations = activations
        self.use_batch_norm = use_batch_norm

        assert len(layers) >= 3, "Invalid layer schema {} for network".format(layers)
        assert (
            len(layers) == len(activations) + 1
        ), "Invalid activation schema {} for network".format(activations)
        assert (
            layers[-2] % 2 == 0
        ), """Last shared layer in dueling architecture should be
        divisible by 2."""

        for i, layer in enumerate(layers[1:-1]):
            self.layers.append(nn.Linear(layers[i], layer))
            self.batch_norm_ops.append(nn.BatchNorm1d(layers[i]))
            gaussian_fill_w_gain(self.layers[i].weight, self.activations[i], layers[i])
            init.constant_(self.layers[i].bias, 0)

        # Split last layer into a value & advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(int(layers[-2] + action_dim), int(layers[-2] / 2)),
            nn.ReLU(),
            nn.Linear(int(layers[-2] / 2), layers[-1]),
        )
        self.value = nn.Sequential(
            nn.Linear(int(layers[-2]), int(layers[-2] / 2)),
            nn.ReLU(),
            nn.Linear(int(layers[-2] / 2), 1),
        )

    def forward(self, input) -> torch.FloatTensor:
        state_dim = self.layers[0].in_features
        state = input[:, :state_dim]
        action = input[:, state_dim:]

        x = state
        for i, activation in enumerate(self.activations[:-1]):
            if self.use_batch_norm:
                x = self.batch_norm_ops[i](x)
            activation_func = getattr(F, activation)
            fc_func = self.layers[i]
            x = fc_func(x) if activation == "linear" else activation_func(fc_func(x))

        value = self.value(x)
        x = torch.cat((x, action), dim=1)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean()
