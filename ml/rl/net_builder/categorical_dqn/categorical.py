#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Dict, List, Type

from ml.rl.models.base import ModelBase
from ml.rl.models.categorical_dqn import CategoricalDQN
from ml.rl.net_builder.categorical_dqn_net_builder import CategoricalDQNNetBuilder
from ml.rl.parameters import NormalizationParameters, param_hash


@dataclass(frozen=True)
class CategoricalConfig:
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    num_atoms: int = 51
    qmin: int = -100
    qmax: int = 200


class Categorical(CategoricalDQNNetBuilder):
    def __init__(self, config: CategoricalConfig):
        super().__init__()
        assert len(config.sizes) == len(config.activations), (
            f"Must have the same numbers of sizes and activations; got: "
            f"{config.sizes}, {config.activations}"
        )
        self.config = config

    @classmethod
    def config_type(cls) -> Type:
        return CategoricalConfig

    def build_q_network(
        self,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        output_dim: int,
    ) -> ModelBase:
        state_dim = self._get_input_dim(state_normalization_parameters)
        return CategoricalDQN(
            state_dim,
            action_dim=output_dim,
            num_atoms=self.config.num_atoms,
            qmin=self.config.qmin,
            qmax=self.config.qmax,
            sizes=self.config.sizes,
            activations=self.config.activations,
            use_batch_norm=False,
            dropout_ratio=0.0,
            use_gpu=False,
        )
