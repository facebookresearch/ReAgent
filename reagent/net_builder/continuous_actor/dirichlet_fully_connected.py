#!/usr/bin/env python3

from typing import List

from reagent.core.dataclasses import dataclass, field
from reagent.models.actor import DirichletFullyConnectedActor
from reagent.models.base import ModelBase
from reagent.net_builder.continuous_actor_net_builder import ContinuousActorNetBuilder
from reagent.parameters import NormalizationData, param_hash
from reagent.preprocessing.identify_types import DO_NOT_PREPROCESS
from reagent.preprocessing.normalization import get_num_output_features


@dataclass
class DirichletFullyConnected(ContinuousActorNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [128, 64])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    use_batch_norm: bool = False

    def __post_init_post_parse__(self):
        super().__init__()
        assert len(self.sizes) == len(self.activations), (
            f"Must have the same numbers of sizes and activations; got: "
            f"{self.sizes}, {self.activations}"
        )

    @property
    def default_action_preprocessing(self) -> str:
        return DO_NOT_PREPROCESS

    def build_actor(
        self,
        state_normalization_data: NormalizationData,
        action_normalization_data: NormalizationData,
    ) -> ModelBase:
        state_dim = get_num_output_features(
            state_normalization_data.dense_normalization_parameters
        )
        action_dim = get_num_output_features(
            action_normalization_data.dense_normalization_parameters
        )
        return DirichletFullyConnectedActor(
            state_dim=state_dim,
            action_dim=action_dim,
            sizes=self.sizes,
            activations=self.activations,
            use_batch_norm=self.use_batch_norm,
        )
