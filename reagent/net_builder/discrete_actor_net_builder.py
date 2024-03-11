#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import abc
from typing import List

import reagent.core.types as rlt
import torch
from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.parameters import NormalizationData
from reagent.models.base import ModelBase
from reagent.prediction.predictor_wrapper import ActorWithPreprocessor
from reagent.preprocessing.preprocessor import Preprocessor


if IS_FB_ENVIRONMENT:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbActorPredictorWrapper as ActorPredictorWrapper,
    )
else:
    from reagent.prediction.predictor_wrapper import ActorPredictorWrapper


class DiscreteActorNetBuilder:
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
        state_feature_config: rlt.ModelFeatureConfig,
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
            actor.cpu_model().eval(), state_preprocessor, state_feature_config
        )
        return ActorPredictorWrapper(
            actor_with_preprocessor, state_feature_config, action_feature_ids
        )
