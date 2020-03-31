#!/usr/bin/env python3

import abc
from typing import Dict, List, Type

import ml.rl.types as rlt
import torch
from ml.rl.core.registry_meta import RegistryMeta
from ml.rl.models.base import ModelBase
from ml.rl.parameters import NormalizationParameters
from ml.rl.prediction.predictor_wrapper import DiscreteDqnWithPreprocessor
from ml.rl.preprocessing.normalization import get_num_output_features
from ml.rl.preprocessing.preprocessor import Preprocessor


try:
    from ml.rl.fb.prediction.fb_predictor_wrapper import (
        FbDiscreteDqnPredictorWrapper as DiscreteDqnPredictorWrapper,
    )
except ImportError:
    from ml.rl.prediction.predictor_wrapper import (  # type: ignore
        DiscreteDqnPredictorWrapper,
    )


class QRDQNNetBuilder(metaclass=RegistryMeta):
    """
    Base class for QRDQN net builder.
    """

    @abc.abstractmethod
    def build_q_network(
        self,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        output_dim: int,
    ) -> ModelBase:
        pass

    def _get_input_dim(
        self, state_normalization_parameters: Dict[int, NormalizationParameters]
    ) -> int:
        return get_num_output_features(state_normalization_parameters)

    def build_serving_module(
        self,
        q_network: ModelBase,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        action_names: List[str],
        state_feature_config: rlt.ModelFeatureConfig,
    ) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        state_preprocessor = Preprocessor(state_normalization_parameters, False)
        dqn_with_preprocessor = DiscreteDqnWithPreprocessor(
            q_network.cpu_model().eval(), state_preprocessor
        )
        return DiscreteDqnPredictorWrapper(
            dqn_with_preprocessor, action_names, state_feature_config
        )
