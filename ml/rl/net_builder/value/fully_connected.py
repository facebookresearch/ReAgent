#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import List, Type

import torch
from ml.rl.models.fully_connected_network import FullyConnectedNetwork
from ml.rl.net_builder.value_net_builder import ValueNetBuilder
from ml.rl.parameters import NormalizationData, param_hash
from ml.rl.preprocessing.normalization import get_num_output_features


@dataclass(frozen=True)
class FullyConnectedConfig:
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    use_layer_norm: bool = False


class FullyConnected(ValueNetBuilder):
    def __init__(self, config: FullyConnectedConfig):
        super().__init__()
        assert len(config.sizes) == len(config.activations), (
            f"Must have the same numbers of sizes and activations; got: "
            f"{config.sizes}, {config.activations}"
        )
        self.config = config

    @classmethod
    def config_type(cls) -> Type:
        return FullyConnectedConfig

    def build_value_network(
        self, state_normalization_data: NormalizationData
    ) -> torch.nn.Module:
        state_dim = get_num_output_features(
            state_normalization_data.dense_normalization_parameters
        )
        return FullyConnectedNetwork(
            [state_dim] + self.config.sizes + [1],
            self.config.activations + ["linear"],
            use_layer_norm=self.config.use_layer_norm,
        )
