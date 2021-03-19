#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from dataclasses import asdict
from typing import List, Optional

import reagent.core.types as rlt
import torch
from reagent.core.configuration import resolve_defaults
from reagent.core.dataclasses import dataclass, field
from reagent.models.base import ModelBase
from torch import nn


EPS = 1e-12


class ScoreCap(nn.Module):
    def __init__(self, cap: float):
        super().__init__()
        self.cap = cap

    def forward(self, input):
        return torch.clip(input, max=self.cap)


@dataclass
class FinalLayer:
    score_cap: Optional[float] = None
    sigmoid: bool = False
    tanh: bool = False

    def __post_init_post_parse__(self):
        assert (
            sum(map(lambda x: int(bool(x)), asdict(self).values())) <= 1
        ), f"More than one option set {self}"

    def get(self):
        if self.score_cap:
            return ScoreCap(self.score_cap)

        if self.sigmoid:
            return nn.Sigmoid()

        if self.tanh:
            return nn.Tanh()

        return nn.Identity()


class MLPScorer(ModelBase):
    """
    Log-space in and out
    """

    @resolve_defaults
    def __init__(
        self,
        input_dim: int,
        layer_sizes: List[int],
        output_dim: int = 1,
        has_user_feat: bool = False,
        final_layer: FinalLayer = field(default_factory=FinalLayer),
    ) -> None:
        super().__init__()
        # Mix Linear layers with ReLU layers, except for the last one.
        inputs = [input_dim] + layer_sizes
        outputs = layer_sizes + [output_dim]
        all_layers = []
        for ind, outd in zip(inputs, outputs):
            all_layers.extend(
                [
                    nn.Linear(ind, outd),
                    nn.ReLU(inplace=True),
                ]
            )
        # drop last relu layer
        all_layers = all_layers[:-1]
        all_layers.append(final_layer.get())
        self.has_user_feat = has_user_feat
        self.mlp = nn.Sequential(*all_layers)

    def forward(self, obs: rlt.FeatureData):
        mlp_input = self._concat_features(obs)
        scores = self.mlp(mlp_input)
        return scores.squeeze(-1)

    def _concat_features(self, obs: rlt.FeatureData):
        if self.has_user_feat:
            return obs.concat_user_doc()
        else:
            return obs.candidate_docs.float_features.float()

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
