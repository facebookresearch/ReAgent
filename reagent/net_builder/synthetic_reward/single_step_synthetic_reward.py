#!/usr/bin/env python3

from typing import List, Optional

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.parameters import NormalizationData, param_hash
from reagent.models.base import ModelBase
from reagent.models.synthetic_reward import SingleStepSyntheticRewardNet
from reagent.net_builder.synthetic_reward_net_builder import SyntheticRewardNetBuilder
from reagent.prediction.predictor_wrapper import (
    ParametricDqnWithPreprocessor,
)
from reagent.preprocessing.normalization import get_num_output_features
from reagent.preprocessing.preprocessor import Preprocessor

if IS_FB_ENVIRONMENT:
    from reagent.fb.prediction.synthetic_reward.single_step_synthetic_reward import (
        FbParametricSingleStepSyntheticRewardPredictorWrapper as ParametricSingleStepSyntheticRewardPredictorWrapper,
    )
else:
    from reagent.prediction.synthetic_reward.single_step_synthetic_reward import (
        ParametricSingleStepSyntheticRewardPredictorWrapper,
    )


@dataclass
class SingleStepSyntheticReward(SyntheticRewardNetBuilder):
    __hash__ = param_hash

    sizes: List[int] = field(default_factory=lambda: [256, 128])
    activations: List[str] = field(default_factory=lambda: ["relu", "relu"])
    last_layer_activation: str = "sigmoid"

    def build_synthetic_reward_network(
        self,
        state_normalization_data: NormalizationData,
        action_normalization_data: Optional[NormalizationData] = None,
        discrete_action_names: Optional[List[str]] = None,
    ) -> ModelBase:
        state_dim = get_num_output_features(
            state_normalization_data.dense_normalization_parameters
        )
        if not discrete_action_names:
            assert action_normalization_data is not None
            action_dim = get_num_output_features(
                action_normalization_data.dense_normalization_parameters
            )
        else:
            action_dim = len(discrete_action_names)
        return SingleStepSyntheticRewardNet(
            state_dim=state_dim,
            action_dim=action_dim,
            sizes=self.sizes,
            activations=self.activations,
            last_layer_activation=self.last_layer_activation,
        )

    def build_serving_module(
        self,
        synthetic_reward_network: ModelBase,
        state_normalization_data: NormalizationData,
        action_normalization_data: Optional[NormalizationData] = None,
        discrete_action_names: Optional[List[str]] = None,
    ) -> torch.nn.Module:
        """
        Returns a TorchScript predictor module
        """
        state_preprocessor = Preprocessor(
            state_normalization_data.dense_normalization_parameters
        )
        if not discrete_action_names:
            assert action_normalization_data is not None
            action_preprocessor = Preprocessor(
                action_normalization_data.dense_normalization_parameters
            )
            synthetic_reward_with_preprocessor = ParametricDqnWithPreprocessor(
                # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a
                #  function.
                synthetic_reward_network.export_mlp().cpu().eval(),
                state_preprocessor,
                action_preprocessor,
            )
            return ParametricSingleStepSyntheticRewardPredictorWrapper(
                synthetic_reward_with_preprocessor
            )
        else:
            raise NotImplementedError(
                "Discrete Single Step Synthetic Reward Predictor has not been implemented"
            )
