#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Optional

import torch
from reagent import types as rlt
from reagent.models.base import ModuleWithDimensions
from reagent.models.dqn import DQNBase, FullyConnectedDQN
from reagent.models.fully_connected_network import FullyConnectedNetwork


logger = logging.getLogger(__name__)


class DuelingQNetwork(DQNBase):
    def __init__(
        self,
        embedding: ModuleWithDimensions,
        action_dim: int,
        feature_extractor: Optional[ModuleWithDimensions] = None,
        quantiles: int = 1,
    ) -> None:
        """
        Dueling Q-Network Architecture: https://arxiv.org/abs/1511.06581
        """
        super().__init__(embedding, action_dim, feature_extractor, quantiles)

        embedding_output_dim = self.embedding.output_dim()

        assert (
            embedding_output_dim % 2 == 0
        ), f"Input to Dueling Q isn't divisible by 2: {embedding_output_dim}"
        # Split last layer into a value & advantage stream
        self.advantage = FullyConnectedDQN(
            action_dim=action_dim,
            embedding=FullyConnectedNetwork(
                layers=[embedding_output_dim, (embedding_output_dim // 2)],
                activations=["leaky_relu"],
            ),
            quantiles=quantiles,
        )
        self.value = FullyConnectedDQN(
            action_dim=1,
            embedding=FullyConnectedNetwork(
                layers=[embedding_output_dim, (embedding_output_dim // 2)],
                activations=["leaky_relu"],
            ),
            quantiles=quantiles,
        )

    def _get_q_values(self, embedding_output: torch.Tensor) -> rlt.AllActionQValues:
        # TODO: Move quantile regression out of DQN so we don't need to wrap and then unwrap here

        wrapped_embedding_output = rlt.PreprocessedState(
            state=rlt.PreprocessedFeatureVector(float_features=embedding_output)
        )
        value = self.value.dist(wrapped_embedding_output).q_values
        raw_advantage = self.advantage.dist(wrapped_embedding_output).q_values
        advantage = raw_advantage - raw_advantage.mean(dim=1, keepdim=True)

        q_values = value + advantage

        if self.quantiles > 1:
            assert q_values.shape[1:] == (
                self.action_dim,
                self.quantiles,
            ), f"Invalid q_values shape: {q_values}"
        else:
            assert (
                q_values.shape[1] == self.action_dim
            ), f"Invalid q_values shape: {q_values}"
        return rlt.AllActionQValues(q_values=q_values)
