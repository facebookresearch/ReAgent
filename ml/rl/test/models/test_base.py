#!/usr/bin/env python3

import unittest
from io import BytesIO

import onnx
import torch
from ml.rl.models.base import ModelBase
from ml.rl.models.types import FeatureVector, StateAction


class Model(ModelBase):
    def __init__(self):
        super(Model, self).__init__()

    def input_prototype(self):
        return StateAction(
            state=FeatureVector(float_features=torch.zeros([1, 4])),
            action=FeatureVector(float_features=torch.zeros([1, 4])),
        )

    def forward(self, sa):
        return sa.state.float_features + sa.action.float_features


class TestBase(unittest.TestCase):
    def test_export(self):
        model = Model()
        buffer = model.export_to_buffer()
        protobuf_model = onnx.load(BytesIO(buffer.getvalue()))
        self.assertEqual(2, len(protobuf_model.graph.input))
        self.assertEqual(1, len(protobuf_model.graph.output))
        self.assertEqual("state:float_features", protobuf_model.graph.input[0].name)
        self.assertEqual("action:float_features", protobuf_model.graph.input[1].name)
