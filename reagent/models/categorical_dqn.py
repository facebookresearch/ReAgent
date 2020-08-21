#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn.functional as F
from reagent import types as rlt
from reagent.models.base import ModelBase


class CategoricalDQN(ModelBase):
    def __init__(
        self,
        distributional_network: ModelBase,
        *,
        qmin: float,
        qmax: float,
        num_atoms: int
    ):
        super().__init__()
        self.distributional_network = distributional_network
        self.support = torch.linspace(qmin, qmax, num_atoms)

    def input_prototype(self):
        return self.distributional_network.input_prototype()

    def forward(self, state: rlt.FeatureData):
        # pyre-fixme[16]: `Tensor` has no attribute `exp`.
        dist = self.log_dist(state).exp()
        q_values = (dist * self.support.to(dist.device)).sum(2)
        return q_values

    def log_dist(self, state: rlt.FeatureData) -> torch.Tensor:
        log_dist = self.distributional_network(state)
        return F.log_softmax(log_dist, -1)
