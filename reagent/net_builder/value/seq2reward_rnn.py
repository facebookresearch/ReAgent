#!/usr/bin/env python3

import torch
from reagent.core.dataclasses import dataclass
from reagent.core.parameters import NormalizationData, param_hash
from reagent.models.seq2reward_model import Seq2RewardNetwork
from reagent.net_builder.value_net_builder import ValueNetBuilder
from reagent.preprocessing.normalization import get_num_output_features


@dataclass
class Seq2RewardNetBuilder(ValueNetBuilder):
    __hash__ = param_hash
    action_dim: int = 2
    num_hiddens: int = 64
    num_hidden_layers: int = 2

    def build_value_network(
        self, state_normalization_data: NormalizationData
    ) -> torch.nn.Module:
        state_dim = get_num_output_features(
            state_normalization_data.dense_normalization_parameters
        )

        return Seq2RewardNetwork(
            state_dim=state_dim,
            action_dim=self.action_dim,
            num_hiddens=self.num_hiddens,
            num_hidden_layers=self.num_hidden_layers,
        )
