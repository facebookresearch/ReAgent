#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from ml.rl import types as rlt
from ml.rl.models.base import ModelBase


logger = logging.getLogger(__name__)


HISTORY_LENGTH = 5


@dataclass
class ExampleSequenceModelOutput:
    value: torch.Tensor


class ExampleSequenceModel(ModelBase):
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = 4
        self.history_length = HISTORY_LENGTH
        self.embedding_size = 20
        self.page_embedding = nn.Embedding(self.embedding_size, self.embedding_dim)
        self.hidden_size = 10
        # ONNX cannot export batch_first=True
        self.gru = nn.GRU(self.embedding_dim, self.hidden_size)
        self.linear = nn.Linear(10 + self.state_dim, 1)

    def input_prototype(self):
        return rlt.PreprocessedState(
            state=rlt.FeatureVector(
                float_features=torch.randn(1, self.state_dim),
                id_list_features={
                    "page_id": (
                        torch.zeros(1, dtype=torch.long),
                        torch.ones(1, dtype=torch.long),
                    )
                },
            )
        )

    def feature_config(self):
        return rlt.ModelFeatureConfig(
            id_mapping_config={
                "page": rlt.IdMapping(ids=list(range(100, 100 + self.embedding_size)))
            },
            id_list_feature_configs=[
                rlt.IdFeatureConfig(
                    name="page_id", feature_id=2002, id_mapping_name="page"
                )
            ],
        )

    def forward(self, state):
        page_embedding = self.page_embedding(state.state.id_list_features["page_id"][1])
        gru_input = page_embedding.unsqueeze(1).transpose(0, 1)
        h_0 = torch.zeros(1, gru_input.shape[1], self.hidden_size)
        gru_output, h_n = self.gru(gru_input, h_0)
        last_gru_output = gru_output[-1, :, :]
        float_features = state.state.float_features
        linear_input = torch.cat((float_features, last_gru_output), dim=1)
        value = self.linear(linear_input)
        return ExampleSequenceModelOutput(value=value)
