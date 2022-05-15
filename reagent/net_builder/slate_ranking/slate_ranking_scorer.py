#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from dataclasses import asdict
from typing import List, Optional

import torch
import torch.nn as nn
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import param_hash
from reagent.models.base import ModelBase
from reagent.models.fully_connected_network import FullyConnectedNetwork
from reagent.models.mlp_scorer import MLPScorer
from reagent.net_builder.slate_ranking_net_builder import SlateRankingNetBuilder


class ScoreCap(nn.Module):
    def __init__(self, cap: float) -> None:
        super().__init__()
        self.cap = cap

    def forward(self, input):
        return torch.clip(input, max=self.cap)


@dataclass
class FinalLayer:
    score_cap: Optional[float] = None
    sigmoid: bool = False
    tanh: bool = False

    def __post_init_post_parse__(self) -> None:
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


@dataclass
class SlateRankingScorer(SlateRankingNetBuilder):
    __hash__ = param_hash

    # For MLP
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    use_batch_norm: bool = False
    min_std: float = 0.0
    dropout_ratio: float = 0.0
    use_layer_norm: bool = False
    normalize_output: bool = False
    orthogonal_init: bool = False

    # For MLP Scorer
    # if disabled, ignores the state features
    has_user_feat: bool = False  # TODO: deprecate
    final_layer: FinalLayer = field(
        default_factory=FinalLayer
    )  # TODO: if score cap not needed, deprecate

    # pyre-fixme[14]: `build_slate_ranking_network` overrides method defined in
    #  `SlateRankingNetBuilder` inconsistently.
    def build_slate_ranking_network(
        self, state_dim, candidate_dim, _candidate_size=None, _slate_size=None
    ) -> ModelBase:
        # pointwise MLP
        input_dim = state_dim + candidate_dim
        output_dim = 1
        layers = [input_dim, *self.hidden_layers, output_dim]
        activations = [
            *self.activations,
            # identity, but we'll add our own final layer
            "linear",
        ]
        mlp = FullyConnectedNetwork(
            layers=layers,
            activations=activations,
            use_batch_norm=self.use_batch_norm,
            min_std=self.min_std,
            dropout_ratio=self.dropout_ratio,
            use_layer_norm=self.use_layer_norm,
            normalize_output=self.normalize_output,
            orthogonal_init=self.orthogonal_init,
        )
        mlp = nn.Sequential(
            *[
                mlp,
                self.final_layer.get(),
            ]
        )
        return MLPScorer(mlp=mlp, has_user_feat=self.has_user_feat)
