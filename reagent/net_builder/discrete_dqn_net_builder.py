#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import abc
from typing import List

import reagent.core.types as rlt
import torch
from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.parameters import NormalizationData
from reagent.models.base import ModelBase
from reagent.prediction.predictor_wrapper import (
    DiscreteDqnWithPreprocessor,
    BinaryDifferenceScorerWithPreprocessor,
)
from reagent.preprocessing.normalization import get_num_output_features
from reagent.preprocessing.preprocessor import Preprocessor


if IS_FB_ENVIRONMENT:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbDiscreteDqnPredictorWrapper as DiscreteDqnPredictorWrapper,
        FbBinaryDifferenceScorerPredictorWrapper as BinaryDifferenceScorerPredictorWrapper,
    )
else:
    from reagent.prediction.predictor_wrapper import (
        DiscreteDqnPredictorWrapper,
        BinaryDifferenceScorerPredictorWrapper,
    )


class DiscreteDQNNetBuilder:
    """
    Base class for discrete DQN net builder.
    """

    @abc.abstractmethod
    def build_q_network(
        self,
        state_feature_config: rlt.ModelFeatureConfig,
        state_normalization_data: NormalizationData,
        output_dim: int,
    ) -> ModelBase:
        pass

    def _get_input_dim(self, state_normalization_data: NormalizationData) -> int:
        return get_num_output_features(
            state_normalization_data.dense_normalization_parameters
        )

    def build_serving_module(
        self,
        q_network: ModelBase,
        state_normalization_data: NormalizationData,
        action_names: List[str],
        state_feature_config: rlt.ModelFeatureConfig,
        predictor_wrapper_type=None,
    ) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        state_preprocessor = Preprocessor(
            state_normalization_data.dense_normalization_parameters, False
        )
        dqn_with_preprocessor = DiscreteDqnWithPreprocessor(
            q_network.cpu_model().eval(), state_preprocessor, state_feature_config
        )
        predictor_wrapper_type = predictor_wrapper_type or DiscreteDqnPredictorWrapper
        return predictor_wrapper_type(
            dqn_with_preprocessor, action_names, state_feature_config
        )

    def build_binary_difference_scorer(
        self,
        q_network: ModelBase,
        state_normalization_data: NormalizationData,
        action_names: List[str],
        state_feature_config: rlt.ModelFeatureConfig,
    ) -> torch.nn.Module:
        """
        Returns softmax(1) - softmax(0)
        """
        assert len(action_names) == 2
        state_preprocessor = Preprocessor(
            state_normalization_data.dense_normalization_parameters, False
        )
        binary_difference_scorer_with_preprocessor = (
            BinaryDifferenceScorerWithPreprocessor(
                q_network.cpu_model().eval(), state_preprocessor, state_feature_config
            )
        )
        return BinaryDifferenceScorerPredictorWrapper(
            binary_difference_scorer_with_preprocessor, state_feature_config
        )
