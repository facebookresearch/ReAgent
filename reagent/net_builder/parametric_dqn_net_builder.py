#!/usr/bin/env python3

import abc

import torch
from reagent.core.registry_meta import RegistryMeta
from reagent.models.base import ModelBase
from reagent.parameters import NormalizationData
from reagent.prediction.predictor_wrapper import ParametricDqnWithPreprocessor
from reagent.preprocessing.preprocessor import Preprocessor


try:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbParametricDqnPredictorWrapper as ParametricDqnPredictorWrapper,
    )
except ImportError:
    from reagent.prediction.predictor_wrapper import ParametricDqnPredictorWrapper


class ParametricDQNNetBuilder(metaclass=RegistryMeta):
    """
    Base class for parametric DQN net builder.
    """

    @abc.abstractmethod
    def build_q_network(
        self,
        state_normalization_data: NormalizationData,
        action_normalization_data: NormalizationData,
        output_dim: int = 1,
    ) -> ModelBase:
        pass

    def build_serving_module(
        self,
        q_network: ModelBase,
        state_normalization_data: NormalizationData,
        action_normalization_data: NormalizationData,
    ) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        state_preprocessor = Preprocessor(
            state_normalization_data.dense_normalization_parameters, False
        )
        action_preprocessor = Preprocessor(
            action_normalization_data.dense_normalization_parameters, False
        )
        dqn_with_preprocessor = ParametricDqnWithPreprocessor(
            q_network.cpu_model().eval(), state_preprocessor, action_preprocessor
        )
        return ParametricDqnPredictorWrapper(
            dqn_with_preprocessor=dqn_with_preprocessor
        )
