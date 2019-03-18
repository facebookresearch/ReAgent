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
class VideoIDFeatures(rlt.IdFeatureBase):
    page_id: rlt.ValueType

    @classmethod
    def get_feature_config(cls) -> Dict[str, rlt.IdFeatureConfig]:
        return {"page_id": rlt.IdFeatureConfig(feature_id=2002, id_mapping_name="page")}


@dataclass
class WatchedVideoSequence(rlt.SequenceFeatureBase):
    id_features: VideoIDFeatures

    @classmethod
    def get_max_length(cls) -> int:
        return HISTORY_LENGTH

    @classmethod
    def get_float_feature_infos(cls) -> List[rlt.FloatFeatureInfo]:
        return [
            rlt.FloatFeatureInfo(name="f{}".format(f_id), feature_id=f_id)
            for f_id in [1001, 1002]
        ]


@dataclass
class SequenceFeatures(rlt.SequenceFeatures):
    """
    The whole class hierarchy can be created dynamically from config.
    Another diff will show this.
    """

    watched_videos: WatchedVideoSequence


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
        self.gru = nn.GRU(
            self.embedding_dim + len(WatchedVideoSequence.get_float_feature_infos()),
            self.hidden_size,
        )
        self.linear = nn.Linear(10 + self.state_dim, 1)

    def input_prototype(self):
        return rlt.StateInput(
            state=rlt.FeatureVector(
                float_features=torch.randn(1, self.state_dim),
                sequence_features=SequenceFeatures.prototype(),
            )
        )

    def feature_config(self):
        return rlt.ModelFeatureConfig(
            id_mapping_config={
                "page": rlt.IdMapping(ids=list(range(100, 100 + self.embedding_size)))
            },
            sequence_features_type=SequenceFeatures,
        )

    def forward(self, state):
        page_embedding = self.page_embedding(
            state.state.sequence_features.watched_videos.id_features.page_id
        )
        gru_input = torch.cat(
            (
                page_embedding,
                state.state.sequence_features.watched_videos.float_features,
            ),
            dim=2,
        ).transpose(0, 1)
        h_0 = torch.zeros(1, gru_input.shape[1], self.hidden_size)
        gru_output, h_n = self.gru(gru_input, h_0)
        last_gru_output = gru_output[-1, :, :]
        float_features = state.state.float_features
        linear_input = torch.cat((float_features, last_gru_output), dim=1)
        value = self.linear(linear_input)
        return ExampleSequenceModelOutput(value=value)
