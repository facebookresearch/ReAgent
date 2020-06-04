#!/usr/bin/env python3

import torch
from reagent.core.dataclasses import dataclass
from reagent.models.seq2reward_model import Seq2RewardNetwork
from reagent.net_builder.value_net_builder import ValueNetBuilder
from reagent.parameters import NormalizationData, param_hash
from reagent.preprocessing.normalization import get_num_output_features


@dataclass
class Seq2RewardNetBuilder(ValueNetBuilder):
    __hash__ = param_hash

    def build_value_network(
        self,
        state_normalization_data: NormalizationData,
        action_normalization_data: NormalizationData,
        num_hiddens: int = 64,
        num_hidden_layers: int = 2,
    ) -> torch.nn.Module:
        state_dim = get_num_output_features(
            state_normalization_data.dense_normalization_parameters
        )
        action_dim = get_num_output_features(
            action_normalization_data.dense_normalization_parameters
        )
        return Seq2RewardNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            num_hiddens=num_hiddens,
            num_hidden_layers=num_hidden_layers,
        )
