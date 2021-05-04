#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import Tuple, List

import torch
from reagent.prediction.predictor_wrapper import (
    ParametricDqnWithPreprocessor,
    ParametricDqnPredictorWrapper,
)


class ParametricSingleStepSyntheticRewardPredictorWrapper(
    ParametricDqnPredictorWrapper
):
    def __init__(
        self,
        synthetic_reward_with_preprocessor: ParametricDqnWithPreprocessor,
    ) -> None:
        super().__init__(synthetic_reward_with_preprocessor)

    # pyre-fixme[56]: Decorator `torch.jit.script_method` could not be resolved in a
    #  global scope.
    @torch.jit.script_method
    def forward(
        self,
        state_with_presence: Tuple[torch.Tensor, torch.Tensor],
        action_with_presence: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[List[str], torch.Tensor]:
        reward = super().forward(state_with_presence, action_with_presence)[1]
        return reward
