#!/usr/bin/env python3

import abc

import torch
from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.registry_meta import RegistryMeta
from reagent.models.base import ModelBase
from reagent.parameters import NormalizationData
from reagent.prediction.predictor_wrapper import (
    ActorWithPreprocessor,
    RankingActorWithPreprocessor,
)
from reagent.preprocessing.postprocessor import Postprocessor
from reagent.preprocessing.preprocessor import Preprocessor


if IS_FB_ENVIRONMENT:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbActorPredictorWrapper as ActorPredictorWrapper,
        FbRankingActorPredictorWrapper as RankingActorPredictorWrapper,
    )
else:
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
        state_normalization_data: NormalizationData,
        action_normalization_data: NormalizationData,
    ) -> ModelBase:
        pass

    def build_serving_module(
        self,
        actor: ModelBase,
        state_normalization_data: NormalizationData,
        action_normalization_data: NormalizationData,
        serve_mean_policy: bool = False,
    ) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """

        state_preprocessor = Preprocessor(
            state_normalization_data.dense_normalization_parameters, use_gpu=False
        )
        postprocessor = Postprocessor(
            action_normalization_data.dense_normalization_parameters, use_gpu=False
        )
        actor_with_preprocessor = ActorWithPreprocessor(
            actor.cpu_model().eval(),
            state_preprocessor,
            postprocessor,
            serve_mean_policy=serve_mean_policy,
        )
        action_features = Preprocessor(
            action_normalization_data.dense_normalization_parameters, use_gpu=False
        ).sorted_features
        return ActorPredictorWrapper(actor_with_preprocessor, action_features)

    def build_ranking_serving_module(
        self,
        actor: ModelBase,
        state_normalization_data: NormalizationData,
        candidate_normalization_data: NormalizationData,
        num_candidates: int,
        action_normalization_data: NormalizationData,
    ) -> torch.nn.Module:
        state_preprocessor = Preprocessor(
            state_normalization_data.dense_normalization_parameters, use_gpu=False
        )
        candidate_preprocessor = Preprocessor(
            candidate_normalization_data.dense_normalization_parameters, use_gpu=False
        )
        postprocessor = Postprocessor(
            action_normalization_data.dense_normalization_parameters, use_gpu=False
        )
        actor_with_preprocessor = RankingActorWithPreprocessor(
            model=actor.cpu_model().eval(),
            state_preprocessor=state_preprocessor,
            candidate_preprocessor=candidate_preprocessor,
            num_candidates=num_candidates,
            action_postprocessor=postprocessor,
        )
        action_features = Preprocessor(
            action_normalization_data.dense_normalization_parameters, use_gpu=False
        ).sorted_features
        return RankingActorPredictorWrapper(actor_with_preprocessor, action_features)
