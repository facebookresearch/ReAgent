#!/usr/bin/env python3

import abc
from typing import List

import torch
from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.parameters import NormalizationData
from reagent.core.registry_meta import RegistryMeta
from reagent.models.base import ModelBase
from reagent.prediction.predictor_wrapper import ActorWithPreprocessor
from reagent.preprocessing.preprocessor import Preprocessor


if IS_FB_ENVIRONMENT:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbActorPredictorWrapper as ActorPredictorWrapper,
    )
else:
    from reagent.prediction.predictor_wrapper import ActorPredictorWrapper


class DiscreteActorNetBuilder(metaclass=RegistryMeta):
    """
    Base class for discrete actor net builder.
    """

    @abc.abstractmethod
    def build_actor(
        self,
        state_normalization_data: NormalizationData,
        num_actions: int,
    ) -> ModelBase:
        pass

    def build_serving_module(
        self,
        actor: ModelBase,
        state_normalization_data: NormalizationData,
        action_feature_ids: List[int],
    ) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """

        state_preprocessor = Preprocessor(
            state_normalization_data.dense_normalization_parameters, use_gpu=False
        )
        actor_with_preprocessor = ActorWithPreprocessor(
            actor.cpu_model().eval(),
            state_preprocessor,
        )
        return ActorPredictorWrapper(actor_with_preprocessor, action_feature_ids)
