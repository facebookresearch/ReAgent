#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import itertools
from dataclasses import field
from typing import List, Optional

import reagent.types as rlt
import torch
from reagent.core.configuration import resolve_defaults
from reagent.models.base import ModelBase
from torch import nn


EPS = 1e-12


class ScoreCap(nn.Module):
    def __init__(self, cap: float):
        super().__init__()
        self.cap = cap

    def forward(self, input):
        return torch.clip(input, max=self.cap)


class MLPScorer(ModelBase):
    @resolve_defaults
    def __init__(
        self,
        input_dim: int,
        layer_sizes: List[int] = field(default_factory=list),  # noqa: B008
        output_dim: int = 1,
        concat: bool = False,
        score_cap: Optional[float] = None,
        log_transform: bool = False,
    ) -> None:
        super().__init__()
        # Mix Linear layers with ReLU layers, except for the last one.
        inputs = [input_dim] + layer_sizes
        outputs = layer_sizes + [output_dim]
        fc_layers = [nn.Linear(ind, outd) for ind, outd in zip(inputs, outputs)]
        relu_layers = [nn.ReLU(inplace=True)] * len(fc_layers)
        all_layers = list(itertools.chain.from_iterable(zip(fc_layers, relu_layers)))[
            :-1
        ]  # drop last relu layer
        if score_cap is not None:
            all_layers.append(ScoreCap(score_cap))
        self.concat = concat
        self.log_transform = log_transform
        self.mlp = nn.Sequential(*all_layers)

    def forward(self, obs):
        if self.log_transform:
            obs = rlt.FeatureData(
                float_features=obs.float_features.clip(EPS).log(),
                candidate_docs=rlt.DocList(
                    float_features=obs.candidate_docs.float_features.clip(EPS).log(),
                ),
            )
        mlp_input = self._concat_features(obs)
        scores = self.mlp(mlp_input)
        return scores.squeeze(-1)

    def _concat_features(self, obs):
        if self.concat:
            return obs.concat_user_doc()
        else:
            return obs.candidate_docs.float_features.float()

    def input_prototype(self):
        # Sample config for input
        batch_size = 2
        state_dim = 5
        num_docs = 3
        candidate_dim = 4
        rlt.FeatureData(
            float_features=torch.randn((batch_size, state_dim)),
            candidate_docs=rlt.DocList(
                float_features=torch.randn(batch_size, num_docs, candidate_dim)
            ),
        )
