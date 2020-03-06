#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Dict, List, Type

from ml.rl import types as rlt
from ml.rl.models.base import ModelBase
from ml.rl.models.dueling_q_network import DuelingQNetwork
from ml.rl.net_builder.discrete_dqn_net_builder import DiscreteDQNNetBuilder
from ml.rl.parameters import NormalizationParameters, param_hash


@dataclass(frozen=True)
class DuelingConfig:
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])


class Dueling(DiscreteDQNNetBuilder):
    def __init__(self, config: DuelingConfig):
        super().__init__()
        assert len(config.sizes) == len(config.activations), (
            f"Must have the same numbers of sizes and activations; got: "
            f"{config.sizes}, {config.activations}"
        )
        self.config = config

    @classmethod
    def config_type(cls) -> Type:
        return DuelingConfig

    def build_q_network(
        self,
        state_feature_config: rlt.ModelFeatureConfig,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        output_dim: int,
    ) -> ModelBase:
        state_dim = self._get_input_dim(state_normalization_parameters)
        return DuelingQNetwork(
            layers=[state_dim] + self.config.sizes + [output_dim],
            activations=self.config.activations + ["linear"],
        )
