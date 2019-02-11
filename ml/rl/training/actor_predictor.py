#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List, Optional

import numpy as np
from caffe2.python import core
from caffe2.python.onnx.workspace import Workspace
from caffe2.python.predictor.predictor_exporter import (
    prepare_prediction_net,
    save_to_db,
)
from ml.rl.training.sandboxed_predictor import SandboxedRLPredictor


logger = logging.getLogger(__name__)


class ActorPredictor(SandboxedRLPredictor):
    def predict(
        self, float_state_features: List[Dict[int, float]]
    ) -> List[Dict[str, float]]:
        self.ws.FeedBlob(
            "input/float_features.lengths",
            np.array([len(e) for e in float_state_features], dtype=np.int32),
        )
        self.ws.FeedBlob(
            "input/float_features.keys",
            np.array(
                [list(e.keys()) for e in float_state_features], dtype=np.int64
            ).flatten(),
        )
        self.ws.FeedBlob(
            "input/float_features.values",
            np.array(
                [list(e.values()) for e in float_state_features], dtype=np.float32
            ).flatten(),
        )

        self.ws.RunNet(self.predict_net)

        output_lengths = self.ws.FetchBlob("output/float_features.lengths")
        output_keys = self.ws.FetchBlob("output/float_features.keys")
        output_values = self.ws.FetchBlob("output/float_features.values")

        first_length = output_lengths[0]
        results = []
        cursor = 0
        for length in output_lengths:
            assert (
                length == first_length
            ), "Number of lengths is not consistent: {}".format(output_lengths)
            result = {}
            for x in range(length):
                result[str(output_keys[cursor + x])] = output_values[cursor + x]
            results.append(result)
            cursor += length

        return results

    def actor_prediction(
        self, float_state_features: List[Dict[int, float]]
    ) -> List[Dict[str, float]]:
        return self.predict(float_state_features)
