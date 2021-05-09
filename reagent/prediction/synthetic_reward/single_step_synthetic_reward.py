#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import Tuple, List

import torch
from reagent.prediction.predictor_wrapper import (
    ParametricDqnWithPreprocessor,
    ParametricDqnPredictorWrapper,
)


class ParametricSingleStepSyntheticRewardPredictorWrapper(torch.jit.ScriptModule):
    def __init__(
        self,
        synthetic_reward_with_preprocessor: ParametricDqnWithPreprocessor,
    ) -> None:
        super().__init__()
        self.wrapper = ParametricDqnPredictorWrapper(synthetic_reward_with_preprocessor)

    @torch.jit.script_method
    def forward(
        self,
        state_with_presence: Tuple[torch.Tensor, torch.Tensor],
        action_with_presence: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        reward = self.wrapper(state_with_presence, action_with_presence)[1]
        return reward
