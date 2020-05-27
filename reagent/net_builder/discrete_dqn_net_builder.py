#!/usr/bin/env python3

import abc
from typing import List

import reagent.types as rlt
import torch
from reagent.core.registry_meta import RegistryMeta
from reagent.models.base import ModelBase
from reagent.parameters import NormalizationData
from reagent.prediction.predictor_wrapper import (
    DiscreteDqnWithPreprocessor,
    DiscreteDqnWithPreprocessorWithIdList,
)
from reagent.preprocessing.normalization import get_num_output_features
from reagent.preprocessing.preprocessor import Preprocessor


try:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbDiscreteDqnPredictorWrapper as DiscreteDqnPredictorWrapper,
        FbDiscreteDqnPredictorWrapperWithIdList as DiscreteDqnPredictorWrapperWithIdList,
    )
except ImportError:
    from reagent.prediction.predictor_wrapper import (
        DiscreteDqnPredictorWrapper,
        DiscreteDqnPredictorWrapperWithIdList,
    )


class DiscreteDQNNetBuilder(metaclass=RegistryMeta):
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
    ) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        state_preprocessor = Preprocessor(
            state_normalization_data.dense_normalization_parameters, False
        )
        dqn_with_preprocessor = DiscreteDqnWithPreprocessor(
            q_network.cpu_model().eval(), state_preprocessor
        )
        return DiscreteDqnPredictorWrapper(
            dqn_with_preprocessor, action_names, state_feature_config
        )


class DiscreteDQNWithIdListNetBuilder(DiscreteDQNNetBuilder):
    """
    Use this in case the model expects ID-list features
    """

    def build_serving_module(
        self,
        q_network: ModelBase,
        state_normalization_data: NormalizationData,
        action_names: List[str],
        state_feature_config: rlt.ModelFeatureConfig,
    ) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        state_preprocessor = Preprocessor(
            state_normalization_data.dense_normalization_parameters, False
        )
        dqn_with_preprocessor = DiscreteDqnWithPreprocessorWithIdList(
            q_network.cpu_model().eval(), state_preprocessor, state_feature_config
        )
        return DiscreteDqnPredictorWrapperWithIdList(
            dqn_with_preprocessor, action_names, state_feature_config
        )
