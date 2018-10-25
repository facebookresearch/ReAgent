#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

from ml.rl.models.parametric_dqn import ParametricDQNWithPreprocessing
from ml.rl.training._parametric_dqn_predictor import _ParametricDQNPredictor


logger = logging.getLogger(__name__)


class RLExporter:
    def __init__(self, dnn, feature_extractor=None, output_transformer=None):
        self.dnn = dnn
        self.feature_extractor = feature_extractor
        self.output_transformer = output_transformer

    def export(self):
        raise NotImplementedError()


class ParametricDQNExporter(RLExporter):
    def __init__(
        self,
        dnn,
        feature_extractor=None,
        output_transformer=None,
        state_preprocessor=None,
        action_preprocessor=None,
    ):
        super(ParametricDQNExporter, self).__init__(
            dnn, feature_extractor, output_transformer
        )
        self.state_preprocessor = state_preprocessor
        self.action_preprocessor = action_preprocessor

    def export(self):
        module_to_export = self.dnn.cpu_model()
        if self.state_preprocessor is not None or self.action_preprocessor is not None:
            module_to_export = ParametricDQNWithPreprocessing(
                module_to_export, self.state_preprocessor, self.action_preprocessor
            )
        pem, ws = module_to_export.get_predictor_export_meta_and_workspace(
            feature_extractor=self.feature_extractor,
            output_transformer=self.output_transformer,
        )
        return _ParametricDQNPredictor(pem, ws)
