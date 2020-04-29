#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from reagent import types as rlt
from reagent.models.base import ModelBase


class StateActionJoiner(ModelBase):
    """
    Concatenating embedding with float features
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    def input_dim(self):
        return self.state_dim + self.action_dim

    def output_dim(self):
        return self.state_dim + self.action_dim

    def input_prototype(self) -> rlt.PreprocessedStateAction:
        return rlt.PreprocessedStateAction.from_tensors(
            state=torch.randn(1, self.state_dim), action=torch.randn(1, self.action_dim)
        )

    def forward(self, input: rlt.PreprocessedStateAction) -> torch.Tensor:
        return torch.cat(
            (input.state.float_features, input.action.float_features), dim=1
        )
