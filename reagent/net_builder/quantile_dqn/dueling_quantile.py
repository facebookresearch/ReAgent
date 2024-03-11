#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

from typing import List

from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import NormalizationData, param_hash
from reagent.models.base import ModelBase
from reagent.models.dueling_q_network import DuelingQNetwork
from reagent.net_builder.quantile_dqn_net_builder import QRDQNNetBuilder


@dataclass
class DuelingQuantile(QRDQNNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])

    def __post_init_post_parse__(self) -> None:
        assert len(self.sizes) == len(self.activations), (
            f"Must have the same numbers of sizes and activations; got: "
            f"{self.sizes}, {self.activations}"
        )

    def build_q_network(
        self,
        state_normalization_data: NormalizationData,
        output_dim: int,
        num_atoms: int,
    ) -> ModelBase:
        state_dim = self._get_input_dim(state_normalization_data)
        return DuelingQNetwork.make_fully_connected(
            state_dim,
            output_dim,
            layers=self.sizes,
            activations=self.activations,
            num_atoms=num_atoms,
        )
