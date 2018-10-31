#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import numpy as np
from caffe2.python import core
from caffe2.python.onnx.workspace import Workspace
from caffe2.python.predictor.predictor_exporter import (
    prepare_prediction_net,
    save_to_db,
)
from ml.rl.training.rl_predictor_pytorch import RLPredictor


logger = logging.getLogger(__name__)


class ActorPredictor(RLPredictor):
    # TODO: Generalizing predictor
    def __init__(self, pem, ws, predict_net=None):
        super(ActorPredictor, self).__init__(
            net=None, init_net=None, parameters=None, int_features=None, ws=ws
        )
        self.pem = pem
        self._predict_net = predict_net

    @property
    def predict_net(self):
        if self._predict_net is None:
            self._predict_net = core.Net(self.pem.predict_net)
            self.ws.CreateNet(self._predict_net)
        return self._predict_net

    def predict(self, float_state_features, int_state_features):
        assert not int_state_features, "Not implemented"
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
                result[output_keys[cursor + x]] = output_values[cursor + x]
            results.append(result)
            cursor += length

        return results

    def actor_prediction(self, float_state_features):
        return self.predict(float_state_features, int_state_features=None)

    def save(self, db_path, db_type):
        # The workspace here is expected to be the Workspace class from ONNX
        with self.ws._ctx:
            save_to_db(db_type, db_path, self.pem)

    @classmethod
    def load(cls, db_path, db_type):
        ws = Workspace()
        with ws._ctx:
            net = prepare_prediction_net(db_path, db_type)
            # TODO: reconstruct pem if so the predictor can be saved back
        return cls(pem=None, ws=ws, predict_net=net)
