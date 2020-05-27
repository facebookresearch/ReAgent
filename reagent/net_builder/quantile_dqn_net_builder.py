#!/usr/bin/env python3

import abc
from typing import List

import reagent.types as rlt
import torch
from reagent.core.registry_meta import RegistryMeta
from reagent.models.base import ModelBase
from reagent.parameters import NormalizationData
from reagent.prediction.predictor_wrapper import DiscreteDqnWithPreprocessor
from reagent.preprocessing.normalization import get_num_output_features
from reagent.preprocessing.preprocessor import Preprocessor


try:
    from reagent.fb.prediction.fb_predictor_wrapper import (
        FbDiscreteDqnPredictorWrapper as DiscreteDqnPredictorWrapper,
    )
except ImportError:
    from reagent.prediction.predictor_wrapper import DiscreteDqnPredictorWrapper


class _Mean(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 3  # type: ignore
        return torch.mean(input, dim=2)


class QRDQNNetBuilder(metaclass=RegistryMeta):
    """
    Base class for QRDQN net builder.
    """

    @abc.abstractmethod
    def build_q_network(
        self,
        state_normalization_data: NormalizationData,
        output_dim: int,
        num_atoms: int,
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
            torch.nn.Sequential(  # type: ignore
                q_network.cpu_model().eval(), _Mean()
            ),
            state_preprocessor,
        )
        return DiscreteDqnPredictorWrapper(
            dqn_with_preprocessor, action_names, state_feature_config
        )
