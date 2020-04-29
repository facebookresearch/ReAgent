#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional

import torch
from reagent import types as rlt
from reagent.models.base import ModelBase, ModuleWithDimensions
from torch.nn.parallel.distributed import DistributedDataParallel


class DQNBase(ModelBase):
    def __init__(
        self,
        embedding: ModuleWithDimensions,
        action_dim: int,
        feature_extractor: Optional[ModuleWithDimensions] = None,
        quantiles: int = 1,
    ):
        """Base class for Action-Value models

        :param embedding: The DNN preceding the output layer
        :param action_dim: Number of actions
        :param feature_extractor: An optional module that transforms the PreprocessedState to a tensor.
            If none, passes PreprocessedState.state.float_features through
        :param quantiles: Number of quantiles for quantile regression
        """
        super().__init__()
        assert action_dim > 0, "action_dim must be > 0, got {}".format(action_dim)
        assert quantiles >= 1
        self.action_dim = action_dim
        self.feature_extractor = feature_extractor
        self.embedding = embedding
        self.quantiles = quantiles

    def input_prototype(self):
        if self.feature_extractor is not None:
            return self.feature_extractor.input_prototype()
        return rlt.PreprocessedState.from_tensor(torch.randn(1, self.input_dim()))

    def forward(self, input) -> rlt.AllActionQValues:
        q_values = self.dist(input).q_values
        if self.quantiles > 1:
            q_values = q_values.reshape((-1, self.action_dim, self.quantiles)).mean(
                dim=2
            )
        return rlt.AllActionQValues(q_values=q_values)

    def dist(self, input: rlt.PreprocessedState) -> rlt.AllActionQValues:
        if self.feature_extractor is not None:
            input_features = self.feature_extractor(input)
        else:
            input_features = input.state.float_features
        embedding_output = self.embedding(input_features)
        return self._get_q_values(embedding_output)

    def _get_q_values(self, embedding_output: torch.Tensor) -> rlt.AllActionQValues:
        raise NotImplementedError()

    @property
    def state_dim(self):
        return self.input_dim()

    def input_dim(self):
        return self.embedding.input_dim()

    def output_dim(self):
        return self.action_dim


class FullyConnectedDQN(DQNBase):
    def __init__(
        self,
        embedding: ModuleWithDimensions,
        action_dim: int,
        feature_extractor: Optional[ModuleWithDimensions] = None,
        quantiles: int = 1,
    ):
        """A DQN with a fully connected layer for q-values
        """
        super().__init__(embedding, action_dim, feature_extractor, quantiles)
        self.layer = torch.nn.Linear(
            self.embedding.output_dim(), self.action_dim * self.quantiles
        )

    def _get_q_values(self, embedding_output: torch.Tensor) -> rlt.AllActionQValues:
        q_values = self.layer(embedding_output)
        if self.quantiles > 1:
            q_values = q_values.reshape((-1, self.action_dim, self.quantiles))
        return rlt.AllActionQValues(q_values=q_values)


class _DistributedDataParallelFullyConnectedDQN(ModelBase):
    def __init__(self, fc_dqn):
        super().__init__()
        self.action_dim = fc_dqn.action_dim
        current_device = torch.cuda.current_device()
        self.data_parallel = DistributedDataParallel(
            fc_dqn.fc, device_ids=[current_device], output_device=current_device
        )
        self.fc_dqn = fc_dqn

    @property
    def state_dim(self):
        return self.input_dim()

    def input_prototype(self):
        return self.fc_dqn.input_prototype()

    def cpu_model(self):
        return self.fc_dqn.cpu_model()

    def forward(self, input):
        q_values = self.data_parallel(input.state.float_features)
        return rlt.AllActionQValues(q_values=q_values)
