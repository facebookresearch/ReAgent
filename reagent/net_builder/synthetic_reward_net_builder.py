#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import abc
from typing import List, Optional

import reagent.core.types as rlt
import torch
from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.parameters import NormalizationData
from reagent.models.base import ModelBase
from reagent.preprocessing.preprocessor import Preprocessor

if IS_FB_ENVIRONMENT:
    from reagent.fb.prediction.synthetic_reward.synthetic_reward_predictor_wrapper import (
        FbSyntheticRewardPredictorWrapper as SyntheticRewardPredictorWrapper,
    )
else:
    from reagent.prediction.synthetic_reward.synthetic_reward_predictor_wrapper import (
        SyntheticRewardPredictorWrapper,
    )


class SyntheticRewardNetBuilder:
    """
    Base class for Synthetic Reward net builder.
    """

    @abc.abstractmethod
    def build_synthetic_reward_network(
        self,
        # dense state features
        state_normalization_data: NormalizationData,
        # dense action features
        action_normalization_data: Optional[NormalizationData] = None,
        discrete_action_names: Optional[List[str]] = None,
        # sparse state features will be read from state_feature_config
        state_feature_config: Optional[rlt.ModelFeatureConfig] = None,
        # sparse action features will be read from action_feature_config
        action_feature_config: Optional[rlt.ModelFeatureConfig] = None,
    ) -> ModelBase:
        pass

    def build_serving_module(
        self,
        seq_len: int,
        synthetic_reward_network: ModelBase,
        state_normalization_data: NormalizationData,
        action_normalization_data: Optional[NormalizationData] = None,
        discrete_action_names: Optional[List[str]] = None,
        # sparse state features will be read from state_feature_config
        state_feature_config: Optional[rlt.ModelFeatureConfig] = None,
        # sparse action features will be read from action_feature_config
        action_feature_config: Optional[rlt.ModelFeatureConfig] = None,
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
            return SyntheticRewardPredictorWrapper(
                seq_len,
                state_preprocessor,
                action_preprocessor,
                synthetic_reward_network.export_mlp().cpu().eval(),
            )
        else:
            # TODO add Discrete Single Step Synthetic Reward Predictor
            return torch.jit.script(torch.nn.Linear(1, 1))
