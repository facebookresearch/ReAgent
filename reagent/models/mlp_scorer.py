#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import reagent.core.types as rlt
import torch
from reagent.models.base import ModelBase


class MLPScorer(ModelBase):
    """
    Log-space in and out
    """

    def __init__(
        self,
        mlp: torch.nn.Module,
        has_user_feat: bool = False,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.has_user_feat = has_user_feat

    def forward(self, obs: rlt.FeatureData):
        mlp_input = obs.get_ranking_state(self.has_user_feat)
        scores = self.mlp(mlp_input)
        return scores.squeeze(-1)

    def input_prototype(self):
        # Sample config for input
        batch_size = 2
        state_dim = 5
        num_docs = 3
        candidate_dim = 4
        return rlt.FeatureData(
            float_features=torch.randn((batch_size, state_dim)),
            candidate_docs=rlt.DocList(
                float_features=torch.randn(batch_size, num_docs, candidate_dim)
            ),
        )
