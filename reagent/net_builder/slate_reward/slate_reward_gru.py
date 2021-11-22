#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import GRUParameters, param_hash
from reagent.models.base import ModelBase
from reagent.models.seq2slate_reward import Seq2SlateGRURewardNet
from reagent.net_builder.slate_reward_net_builder import SlateRewardNetBuilder


@dataclass
class SlateRewardGRU(SlateRewardNetBuilder):
    __hash__ = param_hash

    gru: GRUParameters = field(
        default_factory=lambda: GRUParameters(dim_model=16, num_stacked_layers=2)
    )
    fit_slate_wise_reward: bool = True

    def build_slate_reward_network(
        self, state_dim, candidate_dim, candidate_size, slate_size
    ) -> ModelBase:
        seq2slate_reward_net = Seq2SlateGRURewardNet(
            state_dim=state_dim,
            candidate_dim=candidate_dim,
            num_stacked_layers=self.gru.num_stacked_layers,
            dim_model=self.gru.dim_model,
            max_src_seq_len=candidate_size,
            max_tgt_seq_len=slate_size,
        )
        return seq2slate_reward_net

    @property
    def expect_slate_wise_reward(self):
        return self.fit_slate_wise_reward
