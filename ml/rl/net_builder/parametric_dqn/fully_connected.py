#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Dict, List, Type

from ml.rl.models.base import ModelBase
from ml.rl.models.parametric_dqn import FullyConnectedParametricDQN
from ml.rl.net_builder.parametric_dqn_net_builder import ParametricDQNNetBuilder
from ml.rl.parameters import NormalizationParameters, param_hash
from ml.rl.preprocessing.normalization import get_num_output_features


@dataclass
class FullyConnected(ParametricDQNNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [128, 64])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    use_batch_norm: bool = False
    use_layer_norm: bool = False

    def __post_init__(self):
        super().__init__()
        assert len(self.sizes) == len(self.activations), (
            f"Must have the same numbers of sizes and activations; got: "
            f"{self.sizes}, {self.activations}"
        )

    def build_q_network(
        self,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        action_normalization_parameters: Dict[int, NormalizationParameters],
        output_dim: int = 1,
    ) -> ModelBase:
        state_dim = get_num_output_features(state_normalization_parameters)
        action_dim = get_num_output_features(action_normalization_parameters)
        return FullyConnectedParametricDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            sizes=self.sizes,
            activations=self.activations,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
            output_dim=output_dim,
        )
