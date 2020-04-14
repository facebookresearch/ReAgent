#!/usr/bin/env python3

import abc
from typing import Dict, Type

import torch
from ml.rl.core.registry_meta import RegistryMeta
from ml.rl.models.base import ModelBase
from ml.rl.parameters import NormalizationParameters
from ml.rl.prediction.predictor_wrapper import ParametricDqnWithPreprocessor
from ml.rl.preprocessing.preprocessor import Preprocessor


try:
    from ml.rl.fb.prediction.fb_predictor_wrapper import (
        FbParametricDqnPredictorWrapper as ParametricDqnPredictorWrapper,
    )
except ImportError:
    from ml.rl.prediction.predictor_wrapper import (  # type: ignore
        ParametricDqnPredictorWrapper,
    )


class ParametricDQNNetBuilder(metaclass=RegistryMeta):
    """
    Base class for parametric DQN net builder.
    """

    @abc.abstractmethod
    def build_q_network(
        self,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        action_normalization_parameters: Dict[int, NormalizationParameters],
        output_dim: int = 1,
    ) -> ModelBase:
        pass

    def build_serving_module(
        self,
        q_network: ModelBase,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        action_normalization_parameters: Dict[int, NormalizationParameters],
    ) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        state_preprocessor = Preprocessor(state_normalization_parameters, False)
        action_preprocessor = Preprocessor(action_normalization_parameters, False)
        dqn_with_preprocessor = ParametricDqnWithPreprocessor(
            q_network.cpu_model().eval(), state_preprocessor, action_preprocessor
        )
        return ParametricDqnPredictorWrapper(
            dqn_with_preprocessor=dqn_with_preprocessor
        )
