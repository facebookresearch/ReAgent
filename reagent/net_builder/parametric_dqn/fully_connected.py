#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

from typing import List

from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import NormalizationData, param_hash
from reagent.models.base import ModelBase
from reagent.models.critic import FullyConnectedCritic
from reagent.net_builder.parametric_dqn_net_builder import ParametricDQNNetBuilder
from reagent.preprocessing.normalization import get_num_output_features


@dataclass
class FullyConnected(ParametricDQNNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [128, 64])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    final_activation: str = "linear"

    def __post_init_post_parse__(self) -> None:
        super().__init__()
        assert len(self.sizes) == len(self.activations), (
            f"Must have the same numbers of sizes and activations; got: "
            f"{self.sizes}, {self.activations}"
        )

    def build_q_network(
        self,
        state_normalization_data: NormalizationData,
        action_normalization_data: NormalizationData,
        output_dim: int = 1,
    ) -> ModelBase:
        state_dim = get_num_output_features(
            state_normalization_data.dense_normalization_parameters
        )
        action_dim = get_num_output_features(
            action_normalization_data.dense_normalization_parameters
        )
        return FullyConnectedCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            sizes=self.sizes,
            activations=self.activations,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
            output_dim=output_dim,
            final_activation=self.final_activation,
        )
