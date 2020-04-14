#!/usr/bin/env python3

from typing import List, Type

import torch
from ml.rl.core.dataclasses import dataclass, field
from ml.rl.models.fully_connected_network import FullyConnectedNetwork
from ml.rl.net_builder.value_net_builder import ValueNetBuilder
from ml.rl.parameters import NormalizationData, param_hash
from ml.rl.preprocessing.normalization import get_num_output_features


@dataclass
class FullyConnected(ValueNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    use_layer_norm: bool = False

    def __post_init_post_parse__(self):
        super().__init__()
        assert len(self.sizes) == len(self.activations), (
            f"Must have the same numbers of sizes and activations; got: "
            f"{self.sizes}, {self.activations}"
        )

    def build_value_network(
        self, state_normalization_data: NormalizationData
    ) -> torch.nn.Module:
        state_dim = get_num_output_features(
            state_normalization_data.dense_normalization_parameters
        )
        return FullyConnectedNetwork(
            [state_dim] + self.sizes + [1],
            self.activations + ["linear"],
            use_layer_norm=self.use_layer_norm,
        )
