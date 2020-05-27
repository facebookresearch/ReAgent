#!/usr/bin/env python3

from typing import List

from reagent.core.dataclasses import dataclass, field
from reagent.models.base import ModelBase
from reagent.models.categorical_dqn import CategoricalDQN
from reagent.models.dqn import FullyConnectedDQN
from reagent.net_builder.categorical_dqn_net_builder import CategoricalDQNNetBuilder
from reagent.parameters import NormalizationData, param_hash


@dataclass
class Categorical(CategoricalDQNNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])

    def __post_init_post_parse__(self):
        super().__init__()
        assert len(self.sizes) == len(self.activations), (
            f"Must have the same numbers of sizes and activations; got: "
            f"{self.sizes}, {self.activations}"
        )

    def build_q_network(
        self,
        state_normalization_data: NormalizationData,
        output_dim: int,
        num_atoms: int,
        qmin: int,
        qmax: int,
    ) -> ModelBase:
        state_dim = self._get_input_dim(state_normalization_data)
        distributional_network = FullyConnectedDQN(
            state_dim=state_dim,
            action_dim=output_dim,
            num_atoms=num_atoms,
            sizes=self.sizes,
            activations=self.activations,
            use_batch_norm=False,
            dropout_ratio=0.0,
        )
        return CategoricalDQN(
            distributional_network, qmin=qmin, qmax=qmax, num_atoms=num_atoms
        )
