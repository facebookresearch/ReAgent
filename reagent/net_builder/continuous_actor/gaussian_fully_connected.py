#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import List, Optional

import reagent.models as models
from reagent.core import types as rlt
from reagent.core.dataclasses import dataclass, field
from reagent.core.parameters import NormalizationData, param_hash
from reagent.models.actor import GaussianFullyConnectedActor
from reagent.models.base import ModelBase
from reagent.net_builder.continuous_actor_net_builder import ContinuousActorNetBuilder
from reagent.preprocessing.identify_types import CONTINUOUS_ACTION
from reagent.preprocessing.normalization import get_num_output_features


@dataclass
class GaussianFullyConnected(ContinuousActorNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [128, 64])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    use_l2_normalization: bool = False
    embedding_dim: Optional[int] = None

    def __post_init_post_parse__(self):
        super().__init__()
        assert len(self.sizes) == len(self.activations), (
            f"Must have the same numbers of sizes and activations; got: "
            f"{self.sizes}, {self.activations}"
        )

    @property
    def default_action_preprocessing(self) -> str:
        return CONTINUOUS_ACTION

    def build_actor(
        self,
        state_feature_config: rlt.ModelFeatureConfig,
        state_normalization_data: NormalizationData,
        action_normalization_data: NormalizationData,
    ) -> ModelBase:
        state_dim = get_num_output_features(
            state_normalization_data.dense_normalization_parameters
        )
        action_dim = get_num_output_features(
            action_normalization_data.dense_normalization_parameters
        )
        input_dim = state_dim
        embedding_dim = self.embedding_dim

        embedding_concat = None
        if embedding_dim is not None:
            embedding_concat = models.EmbeddingBagConcat(
                state_dim=state_dim,
                model_feature_config=state_feature_config,
                embedding_dim=embedding_dim,
            )
            input_dim = embedding_concat.output_dim

        gaussian_fc_actor = GaussianFullyConnectedActor(
            state_dim=input_dim,
            action_dim=action_dim,
            sizes=self.sizes,
            activations=self.activations,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
            use_l2_normalization=self.use_l2_normalization,
        )

        if not embedding_dim:
            return gaussian_fc_actor

        assert embedding_concat is not None
        return models.Sequential(  # type: ignore
            embedding_concat,
            rlt.TensorFeatureData(),
            gaussian_fc_actor,
        )
