#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional

import torch
from reagent import types as rlt
from reagent.models.base import ModuleWithDimensions
from reagent.models.dqn import DQNBase, FullyConnectedDQN


class BatchConstrainedDQN(FullyConnectedDQN):
    def __init__(
        self,
        embedding: ModuleWithDimensions,
        action_dim: int,
        imitator_network: DQNBase,
        bcq_drop_threshold: float,
        feature_extractor: Optional[ModuleWithDimensions] = None,
        quantiles: int = 1,
    ):
        super().__init__(embedding, action_dim, feature_extractor, quantiles)
        self.imitator_network = imitator_network
        self.invalid_action_penalty = -1e10
        self.bcq_drop_threshold = bcq_drop_threshold

    def dist(self, input: rlt.PreprocessedState) -> rlt.AllActionQValues:
        raw_q_values = super().dist(input).q_values
        imitator_outputs = self.imitator_network(input).q_values
        imitator_probs = torch.nn.functional.softmax(imitator_outputs, dim=1)
        filter_values = imitator_probs / imitator_probs.max(keepdim=True, dim=1)[0]
        invalid_actions = (filter_values < self.bcq_drop_threshold).float()
        invalid_action_penalty = self.invalid_action_penalty * invalid_actions
        constrained_q_values = raw_q_values + invalid_action_penalty
        return rlt.AllActionQValues(q_values=constrained_q_values)
