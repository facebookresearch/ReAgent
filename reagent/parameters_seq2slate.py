#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from enum import Enum
from typing import Optional

from reagent.core.dataclasses import dataclass


class LearningMethod(Enum):
    TEACHER_FORCING = "teacher_forcing"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    DIFFERENTIABLE_REWARD = "differentiable_reward"
    PAIRWISE_ATTENTION = "pairwise_attention"
    SIMULATION = "simulation"

    @property
    def expect_slate_wise_reward(self):
        return self in (
            LearningMethod.REINFORCEMENT_LEARNING,
            LearningMethod.SIMULATION,
        )


@dataclass(frozen=True)
class RewardClamp:
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None
