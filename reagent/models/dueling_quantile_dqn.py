#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from reagent import types as rlt
from reagent.models.base import ModelBase
from reagent.models.fully_connected_network import gaussian_fill_w_gain
from reagent.tensorboardX import SummaryWriterContext


logger = logging.getLogger(__name__)


class DuelingQuantileDQN(ModelBase):
    def __init__(self, layers, activations, num_atoms=50, use_batch_norm=False) -> None:
        """
        Dueling Q-Network Architecture: https://arxiv.org/abs/1511.06581

        :param layers: List of layer dimensions
        :param activations: List of layer activations
        :param use_batch_norm: bool indicating whether to apply batch normalization
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

        self.num_actions = layers[-1]
        self.num_atoms = num_atoms

        for i, layer in enumerate(layers[1:-1]):
            self.layers.append(nn.Linear(layers[i], layer))
            self.batch_norm_ops.append(nn.BatchNorm1d(layers[i]))
            gaussian_fill_w_gain(self.layers[i].weight, self.activations[i], layers[i])
            init.constant_(self.layers[i].bias, 0)

        # Split last layer into a value & advantage stream
        self.advantage = nn.Sequential(  # type: ignore
            nn.Linear(int(layers[-2]), int(layers[-2] / 2)),
            nn.ReLU(),  # type: ignore
            nn.Linear(int(layers[-2] / 2), layers[-1] * self.num_atoms),
        )
        self.value = nn.Sequential(  # type: ignore
            nn.Linear(int(layers[-2]), int(layers[-2] / 2)),
            nn.ReLU(),  # type: ignore
            nn.Linear(int(layers[-2] / 2), self.num_atoms),
        )
        self._name = "unnamed"

    def input_prototype(self):
        return rlt.PreprocessedState.from_tensor(torch.randn(1, self.state_dim))

    def forward(self, input: rlt.PreprocessedState):
        q_values = self.dist(input).mean(dim=2)
        return rlt.AllActionQValues(q_values=q_values)

    def dist(self, input: rlt.PreprocessedState):
        state = input.state.float_features

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

        value = self.value(x).unsqueeze(dim=1)
        raw_advantage = self.advantage(x).reshape(-1, self.num_actions, self.num_atoms)
        advantage = raw_advantage - raw_advantage.mean(dim=1, keepdim=True)

        q_value = value + advantage

        if SummaryWriterContext._global_step % 1000 == 0:
            SummaryWriterContext.add_histogram(
                "dueling_network/{}/value".format(self._name),
                value.detach().mean(dim=2).cpu(),
            )
            SummaryWriterContext.add_scalar(
                "dueling_network/{}/mean_value".format(self._name),
                value.detach().mean().cpu(),
            )
            SummaryWriterContext.add_histogram(
                "dueling_network/{}/q_value".format(self._name),
                q_value.detach().mean(dim=2).cpu(),
            )
            SummaryWriterContext.add_scalar(
                "dueling_network/{}/mean_q_value".format(self._name),
                q_value.detach().mean().cpu(),
            )
            SummaryWriterContext.add_histogram(
                "dueling_network/{}/raw_advantage".format(self._name),
                raw_advantage.detach().mean(dim=2).cpu(),
            )
            SummaryWriterContext.add_scalar(
                "dueling_network/{}/mean_raw_advantage".format(self._name),
                raw_advantage.detach().mean().cpu(),
            )
            for i in range(advantage.shape[1]):
                a = advantage.detach()[:, i, :].mean(dim=1)
                SummaryWriterContext.add_histogram(
                    "dueling_network/{}/advantage/{}".format(self._name, i), a.cpu()
                )
                SummaryWriterContext.add_scalar(
                    "dueling_network/{}/mean_advantage/{}".format(self._name, i),
                    a.mean().cpu(),
                )

        return q_value
