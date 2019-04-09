#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import os
import tempfile
import unittest
from io import BytesIO
from typing import Any, NamedTuple

import numpy as np
import numpy.testing as npt
import onnx
import torch
import torch.nn as nn
from caffe2.python import core, workspace
from caffe2.python.predictor.predictor_exporter import (
    prepare_prediction_net,
    save_to_db,
)
from ml.rl.models.base import ModelBase
from ml.rl.models.output_transformer import OutputTransformerBase, OutputTransformerNet
from ml.rl.preprocessing.feature_extractor import PredictorFeatureExtractor
from ml.rl.preprocessing.identify_types import CONTINUOUS
from ml.rl.preprocessing.normalization import NormalizationParameters
from ml.rl.test.models.test_utils import check_save_load
from ml.rl.types import FeatureVector, StateAction


logger = logging.getLogger(__name__)


class ModelOutput(NamedTuple):
    # These should be torch.Tensor but the type checking failed when I used it
    sum: Any
    mul: Any
    plus_one: Any
    linear: Any


class Model(ModelBase):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def input_prototype(self):
        return StateAction(
            state=FeatureVector(float_features=torch.randn([1, 4])),
            action=FeatureVector(float_features=torch.randn([1, 4])),
        )

    def forward(self, sa):
        return ModelOutput(
            sa.state.float_features + sa.action.float_features,
            sa.state.float_features * sa.action.float_features,
            sa.state.float_features + 1,
            self.linear(sa.state.float_features),
        )


class TestOutputTransformer(OutputTransformerBase):
    def create_net(self, original_output):
        net = core.Net("output_transformer")
        init_net = core.Net("output_transformer_init")
        action_names = [b"TestAction"]
        action_name_blob = self.create_action_name_blob(init_net, action_names)
        q_value = original_output.linear()
        self.export_q_values(net, q_value, action_names, action_name_blob)
        return OutputTransformerNet(net=net, init_net=init_net)


class TestBase(unittest.TestCase):
    def test_export_to_buffer(self):
        model = Model()
        buffer = model.export_to_buffer()
        protobuf_model = onnx.load(BytesIO(buffer.getvalue()))
        self.assertEqual(4, len(protobuf_model.graph.input))  # 2 inputs + 2 params
        self.assertEqual(4, len(protobuf_model.graph.output))
        self.assertEqual("state:float_features", protobuf_model.graph.input[0].name)
        self.assertEqual("action:float_features", protobuf_model.graph.input[1].name)

    def test_get_predictor_export_meta_and_workspace(self):
        model = Model()

        # 2 params + 1 const
        expected_num_params, expected_num_inputs, expected_num_outputs = 3, 2, 4
        check_save_load(
            self, model, expected_num_params, expected_num_inputs, expected_num_outputs
        )

    def test_get_predictor_export_meta_and_workspace_with_feature_extractor(self):
        model = Model()

        state_normalization_parameters = {
            i: NormalizationParameters(feature_type=CONTINUOUS) for i in range(1, 5)
        }
        action_normalization_parameters = {
            i: NormalizationParameters(feature_type=CONTINUOUS) for i in range(5, 9)
        }

        extractor = PredictorFeatureExtractor(
            state_normalization_parameters=state_normalization_parameters,
            action_normalization_parameters=action_normalization_parameters,
            normalize=False,
        )

        pem, ws = model.get_predictor_export_meta_and_workspace(
            feature_extractor=extractor
        )
        # model has 2 params + 1 const. extractor has 1 const.
        self.assertEqual(4, len(pem.parameters))
        for p in pem.parameters:
            self.assertTrue(ws.HasBlob(p))
        self.assertEqual(3, len(pem.inputs))
        self.assertEqual(4, len(pem.outputs))

        input_prototype = model.input_prototype()

        with tempfile.TemporaryDirectory() as tmpdirname:
            db_path = os.path.join(tmpdirname, "model")
            logger.info("DB path: ", db_path)
            db_type = "minidb"
            with ws._ctx:
                save_to_db(db_type, db_path, pem)

            # Load the model from DB file and run it
            net = prepare_prediction_net(db_path, db_type)

            state_features = input_prototype.state.float_features
            action_features = input_prototype.action.float_features
            float_features_values = (
                torch.cat((state_features, action_features), dim=1).reshape(-1).numpy()
            )
            float_features_keys = np.arange(1, 9)
            float_features_lengths = np.array([8], dtype=np.int32)

            workspace.FeedBlob("input/float_features.keys", float_features_keys)
            workspace.FeedBlob("input/float_features.values", float_features_values)
            workspace.FeedBlob("input/float_features.lengths", float_features_lengths)

            workspace.RunNet(net)
            net_sum = workspace.FetchBlob("sum")
            net_mul = workspace.FetchBlob("mul")
            net_plus_one = workspace.FetchBlob("plus_one")
            net_linear = workspace.FetchBlob("linear")

            model_sum, model_mul, model_plus_one, model_linear = model(input_prototype)

            npt.assert_array_equal(model_sum.numpy(), net_sum)
            npt.assert_array_equal(model_mul.numpy(), net_mul)
            npt.assert_array_equal(model_plus_one.numpy(), net_plus_one)
            npt.assert_allclose(model_linear.detach().numpy(), net_linear, rtol=1e-4)

    def test_get_predictor_export_meta_and_workspace_full(self):
        model = Model()

        state_normalization_parameters = {
            i: NormalizationParameters(feature_type=CONTINUOUS) for i in range(1, 5)
        }
        action_normalization_parameters = {
            i: NormalizationParameters(feature_type=CONTINUOUS) for i in range(5, 9)
        }

        extractor = PredictorFeatureExtractor(
            state_normalization_parameters=state_normalization_parameters,
            action_normalization_parameters=action_normalization_parameters,
            normalize=False,
        )
        output_transformer = TestOutputTransformer()

        pem, ws = model.get_predictor_export_meta_and_workspace(
            feature_extractor=extractor, output_transformer=output_transformer
        )
        # model has 2 params + 1 const. extractor has 1 const. output_transformer has 1 const.
        self.assertEqual(5, len(pem.parameters))
        for p in pem.parameters:
            self.assertTrue(ws.HasBlob(p))
        self.assertEqual(3, len(pem.inputs))
        self.assertEqual(5, len(pem.outputs))
        self.assertEqual(
            {
                "output/string_weighted_multi_categorical_features.lengths",
                "output/string_weighted_multi_categorical_features.keys",
                "output/string_weighted_multi_categorical_features.values.lengths",
                "output/string_weighted_multi_categorical_features.values.keys",
                "output/string_weighted_multi_categorical_features.values.values",
            },
            set(pem.outputs),
        )

        input_prototype = model.input_prototype()

        with tempfile.TemporaryDirectory() as tmpdirname:
            db_path = os.path.join(tmpdirname, "model")
            logger.info("DB path: {}".format(db_path))
            db_type = "minidb"
            with ws._ctx:
                save_to_db(db_type, db_path, pem)

            # Load the model from DB file and run it
            net = prepare_prediction_net(db_path, db_type)

            state_features = input_prototype.state.float_features
            action_features = input_prototype.action.float_features
            float_features_values = (
                torch.cat((state_features, action_features), dim=1).reshape(-1).numpy()
            )
            float_features_keys = np.arange(1, 9)
            float_features_lengths = np.array([8], dtype=np.int32)

            workspace.FeedBlob("input/float_features.keys", float_features_keys)
            workspace.FeedBlob("input/float_features.values", float_features_values)
            workspace.FeedBlob("input/float_features.lengths", float_features_lengths)

            workspace.RunNet(net)

            model_sum, model_mul, model_plus_one, model_linear = model(input_prototype)

            lengths = workspace.FetchBlob(
                "output/string_weighted_multi_categorical_features.lengths"
            )
            keys = workspace.FetchBlob(
                "output/string_weighted_multi_categorical_features.keys"
            )
            values_lengths = workspace.FetchBlob(
                "output/string_weighted_multi_categorical_features.values.lengths"
            )
            values_keys = workspace.FetchBlob(
                "output/string_weighted_multi_categorical_features.values.keys"
            )
            values_values = workspace.FetchBlob(
                "output/string_weighted_multi_categorical_features.values.values"
            )

            N = 1
            npt.assert_array_equal(np.ones(N, dtype=np.int32), lengths)
            npt.assert_array_equal(np.zeros(N, dtype=np.int64), keys)
            npt.assert_array_equal([1] * N, values_lengths)
            npt.assert_array_equal(
                np.array([b"TestAction"], dtype=np.object), values_keys
            )
            npt.assert_array_equal(
                model_linear.detach().numpy().reshape(-1), values_values
            )
