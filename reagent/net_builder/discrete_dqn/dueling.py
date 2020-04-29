#!/usr/bin/env python3

from typing import Dict, List, Type

from reagent import types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.models.base import ModelBase
from reagent.models.dueling_q_network import DuelingQNetwork
from reagent.models.fully_connected_network import FullyConnectedNetwork
from reagent.net_builder.discrete_dqn_net_builder import DiscreteDQNNetBuilder
from reagent.parameters import NormalizationParameters, param_hash


@dataclass
class Dueling(DiscreteDQNNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    quantiles: int = 1

    def __post_init_post_parse__(self):
        assert len(self.sizes) == len(self.activations), (
            f"Must have the same numbers of sizes and activations; got: "
            f"{self.sizes}, {self.activations}"
        )

    def build_q_network(
        self,
        state_feature_config: rlt.ModelFeatureConfig,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        output_dim: int,
    ) -> ModelBase:
        state_dim = self._get_input_dim(state_normalization_parameters)
        return DuelingQNetwork(
            action_dim=output_dim,
            embedding=FullyConnectedNetwork(
                layers=[state_dim] + self.sizes, activations=self.activations
            ),
            quantiles=self.quantiles,
        )
