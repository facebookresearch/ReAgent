#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from enum import Enum


class LearningMethod(Enum):
    TEACHER_FORCING = "teacher_forcing"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    DIFFERENTIABLE_REWARD = "differentiable_reward"
    PAIRWISE_ATTENTION = "pairwise_attention"
