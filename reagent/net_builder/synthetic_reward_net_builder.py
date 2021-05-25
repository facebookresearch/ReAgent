#!/usr/bin/env python3

import abc
from typing import List, Optional

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
        state_normalization_data: NormalizationData,
        action_normalization_data: Optional[NormalizationData] = None,
        discrete_action_names: Optional[List[str]] = None,
    ) -> ModelBase:
        pass

    def build_serving_module(
        self,
        seq_len: int,
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
            return SyntheticRewardPredictorWrapper(
                seq_len,
                state_preprocessor,
                action_preprocessor,
                # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a
                #  function.
                synthetic_reward_network.export_mlp().cpu().eval(),
            )
        else:
            # TODO add Discrete Single Step Synthetic Reward Predictor
            return torch.jit.script(torch.nn.Linear(1, 1))
