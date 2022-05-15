#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import param_hash, TransformerParameters
from reagent.model_utils.seq2slate_utils import Seq2SlateOutputArch
from reagent.models.base import ModelBase
from reagent.models.seq2slate import Seq2SlateTransformerNet
from reagent.net_builder.slate_ranking_net_builder import SlateRankingNetBuilder


@dataclass
class SlateRankingTransformer(SlateRankingNetBuilder):
    __hash__ = param_hash

    output_arch: Seq2SlateOutputArch = Seq2SlateOutputArch.AUTOREGRESSIVE
    temperature: float = 1.0
    transformer: TransformerParameters = field(
        default_factory=lambda: TransformerParameters(
            num_heads=2, dim_model=16, dim_feedforward=16, num_stacked_layers=2
        )
    )

    def build_slate_ranking_network(
        self, state_dim, candidate_dim, candidate_size, slate_size
    ) -> ModelBase:
        return Seq2SlateTransformerNet(
            state_dim=state_dim,
            candidate_dim=candidate_dim,
            num_stacked_layers=self.transformer.num_stacked_layers,
            num_heads=self.transformer.num_heads,
            dim_model=self.transformer.dim_model,
            dim_feedforward=self.transformer.dim_feedforward,
            max_src_seq_len=candidate_size,
            max_tgt_seq_len=slate_size,
            output_arch=self.output_arch,
            temperature=self.temperature,
            state_embed_dim=self.transformer.state_embed_dim,
        )
