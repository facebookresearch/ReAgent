#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import List

import reagent.models as models
from reagent.core import types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import NormalizationData, param_hash
from reagent.net_builder.discrete_dqn_net_builder import DiscreteDQNNetBuilder


@dataclass
class FullyConnectedWithEmbedding(DiscreteDQNNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    dropout_ratio: float = 0.0

    def __post_init_post_parse__(self):
        super().__init__()
        assert len(self.sizes) == len(self.activations), (
            f"Must have the same numbers of sizes and activations; got: "
            f"{self.sizes}, {self.activations}"
        )

    def build_q_network(
        self,
        state_feature_config: rlt.ModelFeatureConfig,
        state_normalization_data: NormalizationData,
        output_dim: int,
    ) -> models.ModelBase:
        state_dense_dim = self._get_input_dim(state_normalization_data)
        embedding_concat = models.EmbeddingBagConcat(
            state_dense_dim=state_dense_dim,
            model_feature_config=state_feature_config,
        )
        return models.Sequential(  # type: ignore
            embedding_concat,
            rlt.TensorFeatureData(),
            models.FullyConnectedDQN(
                embedding_concat.output_dim,
                action_dim=output_dim,
                sizes=self.sizes,
                activations=self.activations,
                dropout_ratio=self.dropout_ratio,
            ),
        )
