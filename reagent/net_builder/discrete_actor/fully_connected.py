#!/usr/bin/env python3

from typing import List, Optional

from reagent.core.dataclasses import dataclass, field
from reagent.models.actor import FullyConnectedActor
from reagent.models.base import ModelBase
from reagent.net_builder.discrete_actor_net_builder import DiscreteActorNetBuilder
from reagent.parameters import NormalizationData, param_hash
from reagent.preprocessing.normalization import get_num_output_features


@dataclass
class FullyConnected(DiscreteActorNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [128, 64])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    action_activation: str = "tanh"
    exploration_variance: Optional[float] = None

    def __post_init_post_parse__(self):
        super().__init__()
        assert len(self.sizes) == len(self.activations), (
            f"Must have the same numbers of sizes and activations; got: "
            f"{self.sizes}, {self.activations}"
        )

    def build_actor(
        self,
        state_normalization_data: NormalizationData,
        num_actions: int,
    ) -> ModelBase:
        state_dim = get_num_output_features(
            state_normalization_data.dense_normalization_parameters
        )
        return FullyConnectedActor(
            state_dim=state_dim,
            action_dim=num_actions,
            sizes=self.sizes,
            activations=self.activations,
            use_batch_norm=self.use_batch_norm,
            action_activation=self.action_activation,
            exploration_variance=self.exploration_variance,
        )
