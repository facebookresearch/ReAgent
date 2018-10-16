#!/usr/bin/env python3

import logging

from caffe2.python import core
from ml.rl.training.rl_predictor_pytorch import RLPredictor


logger = logging.getLogger(__name__)


class _ParametricDQNPredictor(RLPredictor):
    def __init__(self, pem, ws):
        super(_ParametricDQNPredictor, self).__init__(
            net=None, init_net=None, parameters=None, int_features=None, ws=ws
        )
        self.pem = pem
        self._predict_net = None

    @property
    def predict_net(self):
        if self._predict_net is None:
            self._predict_net = core.Net(self.pem.predict_net)
            self.ws.CreateNet(self._predict_net)
        return self._predict_net

    def predict(self, float_state_features, int_state_features, actions):
        assert not int_state_features, "Not implemented"

        float_examples = []
        for i in range(len(float_state_features)):
            float_examples.append({**float_state_features[i], **actions[i]})

        return super(_ParametricDQNPredictor, self).predict(float_examples)

    def save(self, db_path, db_type):
        raise NotImplementedError

    @classmethod
    def export(cls, q_network, feature_extractor=None, output_transformer=None):
        pem, ws = q_network.get_predictor_export_meta_and_workspace(
            feature_extractor=feature_extractor, output_transformer=output_transformer
        )
        return cls(pem, ws)
