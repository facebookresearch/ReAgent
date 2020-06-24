#!/usr/bin/env python3

from reagent.core.dataclasses import dataclass, field
from reagent.models.base import ModelBase
from reagent.models.seq2slate_reward import Seq2SlateRewardNet
from reagent.net_builder.slate_reward_net_builder import SlateRewardNetBuilder
from reagent.parameters import TransformerParameters, param_hash


@dataclass
class SlateRewardTransformer(SlateRewardNetBuilder):
    __hash__ = param_hash

    transformer: TransformerParameters = field(
        default_factory=lambda: TransformerParameters(
            num_heads=2, dim_model=16, dim_feedforward=16, num_stacked_layers=2
        )
    )
    fit_slate_wise_reward: bool = True

    def build_slate_reward_network(
        self, state_dim, candidate_dim, candidate_size, slate_size
    ) -> ModelBase:
        seq2slate_reward_net = Seq2SlateRewardNet(
            state_dim=state_dim,
            candidate_dim=candidate_dim,
            num_stacked_layers=self.transformer.num_stacked_layers,
            num_heads=self.transformer.num_heads,
            dim_model=self.transformer.dim_model,
            dim_feedforward=self.transformer.dim_feedforward,
            max_src_seq_len=candidate_size,
            max_tgt_seq_len=slate_size,
        )
        return seq2slate_reward_net

    @property
    def expect_slate_wise_reward(self):
        return self.fit_slate_wise_reward
