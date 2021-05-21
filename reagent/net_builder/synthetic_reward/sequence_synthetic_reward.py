#!/usr/bin/env python3

from typing import List, Optional

from reagent.core.dataclasses import dataclass
from reagent.core.parameters import NormalizationData, param_hash
from reagent.models.base import ModelBase
from reagent.models.synthetic_reward import (
    SequenceSyntheticRewardNet,
    SyntheticRewardNet,
)
from reagent.net_builder.synthetic_reward_net_builder import SyntheticRewardNetBuilder
from reagent.preprocessing.normalization import get_num_output_features


@dataclass
class SequenceSyntheticReward(SyntheticRewardNetBuilder):
    __hash__ = param_hash

    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = False
    last_layer_activation: str = "sigmoid"

    def build_synthetic_reward_network(
        self,
        state_normalization_data: NormalizationData,
        action_normalization_data: Optional[NormalizationData] = None,
        discrete_action_names: Optional[List[str]] = None,
    ) -> ModelBase:
        state_dim = get_num_output_features(
            state_normalization_data.dense_normalization_parameters
        )
        if not discrete_action_names:
            assert action_normalization_data is not None
            action_dim = get_num_output_features(
                action_normalization_data.dense_normalization_parameters
            )
        else:
            action_dim = len(discrete_action_names)
        net = SequenceSyntheticRewardNet(
            state_dim=state_dim,
            action_dim=action_dim,
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_num_layers=self.lstm_num_layers,
            lstm_bidirectional=self.lstm_bidirectional,
            last_layer_activation=self.last_layer_activation,
        )
        return SyntheticRewardNet(net=net)
