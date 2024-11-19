#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging
from typing import List, Tuple

import torch
from reagent.core import types as rlt
from reagent.prediction.predictor_wrapper import DiscreteDqnWithPreprocessor

logger = logging.getLogger(__name__)


class BanditRewardNetPredictorWrapper(torch.jit.ScriptModule):
    def __init__(
        self,
        reward_model_with_preprocessor: DiscreteDqnWithPreprocessor,
        action_names: List[str],
        state_feature_config: rlt.ModelFeatureConfig,
    ) -> None:
        super().__init__()
        self.reward_model_with_preprocessor = torch.jit.trace(
            reward_model_with_preprocessor,
            reward_model_with_preprocessor.input_prototype(),
        )
        self.action_names = torch.jit.Attribute(action_names, List[str])

    @torch.jit.script_method
    def forward(
        self, state: rlt.ServingFeatureData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reward_predictions = self.reward_model_with_preprocessor(state)
        num_examples = reward_predictions.size()[0]
        num_actions = len(self.action_names)
        assert (
            reward_predictions.shape
            == (
                num_examples,
                num_actions,
            )
        ), f"Invalid shape {reward_predictions.shape} != ({num_examples}, {num_actions})"
        mask = torch.ones_like(reward_predictions, dtype=torch.uint8)
        return (reward_predictions, mask)
