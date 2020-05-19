#!/usr/bin/env python3

import abc
from typing import Dict

import torch
from reagent.core.registry_meta import RegistryMeta
from reagent.models.base import ModelBase
from reagent.parameters import NormalizationParameters
from reagent.prediction.predictor_wrapper import ActorWithPreprocessor
from reagent.preprocessing.postprocessor import Postprocessor
from reagent.preprocessing.preprocessor import Preprocessor


try:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbActorPredictorWrapper as ActorPredictorWrapper,
    )
except ImportError:
    from reagent.prediction.predictor_wrapper import ActorPredictorWrapper


class ContinuousActorNetBuilder(metaclass=RegistryMeta):
    """
    Base class for continuous actor net builder.
    """

    @property
    @abc.abstractmethod
    def default_action_preprocessing(self) -> str:
        pass

    @abc.abstractmethod
    def build_actor(
        self,
        state_normalization: Dict[int, NormalizationParameters],
        action_normalization: Dict[int, NormalizationParameters],
    ) -> ModelBase:
        pass

    def build_serving_module(
        self,
        actor: ModelBase,
        state_normalization: Dict[int, NormalizationParameters],
        action_normalization: Dict[int, NormalizationParameters],
    ) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """

        state_preprocessor = Preprocessor(state_normalization, use_gpu=False)
        postprocessor = Postprocessor(action_normalization, use_gpu=False)
        actor_with_preprocessor = ActorWithPreprocessor(
            actor.cpu_model().eval(), state_preprocessor, postprocessor
        )
        action_features = Preprocessor(
            action_normalization, use_gpu=False
        ).sorted_features
        return ActorPredictorWrapper(actor_with_preprocessor, action_features)
