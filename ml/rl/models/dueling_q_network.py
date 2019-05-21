#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ml.rl import types as rlt
from ml.rl.models.base import ModelBase
from ml.rl.models.fully_connected_network import gaussian_fill_w_gain
from ml.rl.tensorboardX import SummaryWriterContext


logger = logging.getLogger(__name__)


class DuelingQNetwork(ModelBase):
    def __init__(self, layers, activations, use_batch_norm=False, action_dim=0) -> None:
        """
        Dueling Q-Network Architecture: https://arxiv.org/abs/1511.06581

        :param layers: List of layer dimensions
        :param activations: List of layer activations
        :param use_batch_norm: bool indicating whether to apply batch normalization
        :param action_dim: if !=0 use parametric dueling DQN, else standard dueling DQN
        """
        super().__init__()
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

        self.state_dim = layers[0]
        self.action_dim = action_dim

        for i, layer in enumerate(layers[1:-1]):
            self.layers.append(nn.Linear(layers[i], layer))
            self.batch_norm_ops.append(nn.BatchNorm1d(layers[i]))
            gaussian_fill_w_gain(self.layers[i].weight, self.activations[i], layers[i])
            init.constant_(self.layers[i].bias, 0)

        self.parametric_action = action_dim > 0
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
        self._name = "unnamed"

    def input_prototype(self):
        if self.parametric_action:
            return rlt.StateAction(
                state=rlt.FeatureVector(float_features=torch.randn(1, self.state_dim)),
                action=rlt.FeatureVector(
                    float_features=torch.randn(1, self.action_dim)
                ),
            )
        else:
            return rlt.StateInput(
                state=rlt.FeatureVector(float_features=torch.randn(1, self.state_dim))
            )

    def forward(self, input) -> torch.FloatTensor:
        output_tensor = False
        if isinstance(input, torch.Tensor):
            # Maintaining backward compatibility for a bit
            state_dim = self.layers[0].in_features
            state = input[:, :state_dim]
            action = input[:, state_dim:]
            output_tensor = True
        elif self.parametric_action:
            state = input.state.float_features
            action = input.action.float_features
        else:
            state = input.state.float_features
            action = None

        x = state
        for i, activation in enumerate(self.activations[:-1]):
            if self.use_batch_norm:
                x = self.batch_norm_ops[i](x)

            x = self.layers[i](x)
            if activation == "linear":
                continue
            elif activation == "tanh":
                activation_func = torch.tanh
            else:
                activation_func = getattr(F, activation)
            x = activation_func(x)

        value = self.value(x)
        if action is not None:
            x = torch.cat((x, action), dim=1)
        raw_advantage = self.advantage(x)
        if self.parametric_action:
            advantage = raw_advantage
        else:
            advantage = raw_advantage - raw_advantage.mean(dim=1, keepdim=True)

        q_value = value + advantage

        if SummaryWriterContext._global_step % 1000 == 0:
            SummaryWriterContext.add_histogram(
                "dueling_network/{}/value".format(self._name), value.detach().cpu()
            )
            SummaryWriterContext.add_scalar(
                "dueling_network/{}/mean_value".format(self._name),
                value.detach().mean().cpu(),
            )
            SummaryWriterContext.add_histogram(
                "dueling_network/{}/q_value".format(self._name), q_value.detach().cpu()
            )
            SummaryWriterContext.add_scalar(
                "dueling_network/{}/mean_q_value".format(self._name),
                q_value.detach().mean().cpu(),
            )
            SummaryWriterContext.add_histogram(
                "dueling_network/{}/raw_advantage".format(self._name),
                raw_advantage.detach().cpu(),
            )
            SummaryWriterContext.add_scalar(
                "dueling_network/{}/mean_raw_advantage".format(self._name),
                raw_advantage.detach().mean().cpu(),
            )
            if not self.parametric_action:
                for i in range(advantage.shape[1]):
                    a = advantage.detach()[:, i]
                    SummaryWriterContext.add_histogram(
                        "dueling_network/{}/advatage/{}".format(self._name, i), a.cpu()
                    )
                    SummaryWriterContext.add_scalar(
                        "dueling_network/{}/mean_advatage/{}".format(self._name, i),
                        a.mean().cpu(),
                    )

        if output_tensor:
            return q_value
        elif self.parametric_action:
            return rlt.SingleQValue(q_value=q_value)
        else:
            return rlt.AllActionQValues(q_values=q_value)
