#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import List, Type

from ml.rl.models.actor import DirichletFullyConnectedActor
from ml.rl.models.base import ModelBase
from ml.rl.net_builder.continuous_actor_net_builder import ContinuousActorNetBuilder
from ml.rl.parameters import NormalizationData, param_hash
from ml.rl.preprocessing.identify_types import DO_NOT_PREPROCESS
from ml.rl.preprocessing.normalization import get_num_output_features


@dataclass(frozen=True)
class DirichletFullyConnectedConfig:
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [128, 64])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    use_batch_norm: bool = False


class DirichletFullyConnected(ContinuousActorNetBuilder):
    def __init__(self, config: DirichletFullyConnectedConfig):
        super().__init__()
        assert len(config.sizes) == len(config.activations), (
            f"Must have the same numbers of sizes and activations; got: "
            f"{config.sizes}, {config.activations}"
        )
        self.config = config

    @classmethod
    def config_type(cls) -> Type:
        return DirichletFullyConnectedConfig

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
            sizes=self.config.sizes,
            activations=self.config.activations,
            use_batch_norm=self.config.use_batch_norm,
        )
