#!/usr/bin/env python3

import abc
from typing import Type

import torch
from ml.rl.core.registry_meta import RegistryMeta
from ml.rl.models.base import ModelBase
from ml.rl.parameters import NormalizationData
from ml.rl.prediction.predictor_wrapper import ActorWithPreprocessor
from ml.rl.preprocessing.postprocessor import Postprocessor
from ml.rl.preprocessing.preprocessor import Preprocessor


try:
    from ml.rl.fb.prediction.fb_predictor_wrapper import (
        FbActorPredictorWrapper as ActorPredictorWrapper,
    )
except ImportError:
    from ml.rl.prediction.predictor_wrapper import ActorPredictorWrapper  # type: ignore


class ContinuousActorNetBuilder(metaclass=RegistryMeta):
    """
    Base class for continuous actor net builder.
    """

    @classmethod
    @abc.abstractmethod
    def config_type(cls) -> Type:
        """
        Return the config type. Must be conforming to Flow python 3 type API
        """
        pass

    @property
    @abc.abstractmethod
    def default_action_preprocessing(self) -> str:
        pass

    @abc.abstractmethod
    def build_actor(
        self,
        state_normalization_data: NormalizationData,
        action_normalization_data: NormalizationData,
    ) -> ModelBase:
        pass

    def build_serving_module(
        self,
        actor: ModelBase,
        state_normalization_data: NormalizationData,
        action_normalization_data: NormalizationData,
    ) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        state_normalization_parameters = (
            state_normalization_data.dense_normalization_parameters
        )
        action_normalization_parameters = (
            action_normalization_data.dense_normalization_parameters
        )
        assert state_normalization_parameters is not None
        assert action_normalization_parameters is not None

        state_preprocessor = Preprocessor(state_normalization_parameters, use_gpu=False)
        postprocessor = Postprocessor(action_normalization_parameters, use_gpu=False)
        actor_with_preprocessor = ActorWithPreprocessor(
            actor.cpu_model().eval(), state_preprocessor, postprocessor
        )
        action_features = Preprocessor(
            action_normalization_parameters, use_gpu=False
        ).sorted_features
        return ActorPredictorWrapper(actor_with_preprocessor, action_features)
