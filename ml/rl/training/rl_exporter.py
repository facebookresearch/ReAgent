#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

from ml.rl.models.actor import ActorWithPreprocessing
from ml.rl.models.output_transformer import (
    ActorOutputTransformer,
    ParametricActionOutputTransformer,
)
from ml.rl.models.parametric_dqn import ParametricDQNWithPreprocessing
from ml.rl.preprocessing.feature_extractor import PredictorFeatureExtractor
from ml.rl.preprocessing.normalization import get_action_output_parameters
from ml.rl.training.actor_predictor import ActorPredictor
from ml.rl.training.dqn_predictor import DQNPredictor
from ml.rl.training.parametric_dqn_predictor import ParametricDQNPredictor


logger = logging.getLogger(__name__)


class RLExporter:
    def __init__(self, dnn, feature_extractor=None, output_transformer=None):
        self.dnn = dnn
        self.feature_extractor = feature_extractor
        self.output_transformer = output_transformer

    def export(self):
        raise NotImplementedError()


class SandboxedRLExporter(RLExporter):
    def __init__(
        self,
        dnn,
        predictor_class,
        preprocessing_class,
        feature_extractor=None,
        output_transformer=None,
        state_preprocessor=None,
        action_preprocessor=None,
        **kwargs,
    ):
        super().__init__(dnn, feature_extractor, output_transformer)
        self.state_preprocessor = state_preprocessor
        self.action_preprocessor = action_preprocessor
        self.predictor_class = predictor_class
        self.preprocessing_class = preprocessing_class
        self.kwargs = kwargs

    def export(self):
        module_to_export = self.dnn.cpu_model()
        if self.action_preprocessor:
            module_to_export = self.preprocessing_class(
                module_to_export, self.state_preprocessor, self.action_preprocessor
            )
        elif self.state_preprocessor:
            module_to_export = self.preprocessing_class(
                module_to_export, self.state_preprocessor
            )
        pem, ws = module_to_export.get_predictor_export_meta_and_workspace(
            feature_extractor=self.feature_extractor,
            output_transformer=self.output_transformer,
        )
        return self.predictor_class(pem, ws, **self.kwargs)


class ParametricDQNExporter(SandboxedRLExporter):
    def __init__(
        self,
        dnn,
        feature_extractor=None,
        output_transformer=None,
        state_preprocessor=None,
        action_preprocessor=None,
    ):
        super().__init__(
            dnn,
            ParametricDQNPredictor,
            ParametricDQNWithPreprocessing,
            feature_extractor,
            output_transformer,
            state_preprocessor,
            action_preprocessor,
        )

    @classmethod
    def from_state_action_normalization(
        cls,
        dnn,
        state_normalization,
        action_normalization,
        state_preprocessor=None,
        action_preprocessor=None,
        **kwargs,
    ):
        return cls(
            dnn=dnn,
            feature_extractor=PredictorFeatureExtractor(
                state_normalization_parameters=state_normalization,
                action_normalization_parameters=action_normalization,
            ),
            output_transformer=ParametricActionOutputTransformer(),
            state_preprocessor=state_preprocessor,
            action_preprocessor=action_preprocessor,
            **kwargs,
        )


class DQNExporter(SandboxedRLExporter):
    def __init__(
        self,
        dnn,
        feature_extractor=None,
        output_transformer=None,
        state_preprocessor=None,
        predictor_class=DQNPredictor,
        preprocessing_class=None,
        **kwargs,
    ):
        super().__init__(
            dnn,
            predictor_class,
            preprocessing_class,
            feature_extractor,
            output_transformer,
            state_preprocessor,
            action_preprocessor=None,
            **kwargs,
        )


class ActorExporter(SandboxedRLExporter):
    def __init__(
        self,
        dnn,
        feature_extractor=None,
        output_transformer=None,
        state_preprocessor=None,
        predictor_class=ActorPredictor,
        **kwargs,
    ):
        super().__init__(
            dnn,
            predictor_class,
            ActorWithPreprocessing,
            feature_extractor,
            output_transformer,
            state_preprocessor,
            action_preprocessor=None,
            **kwargs,
        )

    @classmethod
    def from_state_action_normalization(
        cls,
        dnn,
        state_normalization,
        action_normalization,
        state_preprocessor=None,
        predictor_class=ActorPredictor,
        **kwargs,
    ):
        feature_extractor = PredictorFeatureExtractor(
            state_normalization_parameters=state_normalization
        )

        action_feature_ids, serving_min_scale, serving_max_scale = get_action_output_parameters(  # noqa B950
            action_normalization
        )
        output_transformer = ActorOutputTransformer(
            action_feature_ids, serving_max_scale, serving_min_scale
        )
        return cls(
            dnn,
            feature_extractor=feature_extractor,
            output_transformer=output_transformer,
            state_preprocessor=state_preprocessor,
            predictor_class=predictor_class,
            **kwargs,
        )
