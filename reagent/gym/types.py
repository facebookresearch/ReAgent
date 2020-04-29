#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Please DO NOT import gym in here. We might have installation without gym depending on
# this module for typing

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

import reagent.types as rlt
import torch


class Sampler(ABC):
    """Given scores, select the action."""

    @abstractmethod
    def sample_action(
        self, scores: Any, possible_action_mask: Optional[Any]
    ) -> rlt.ActorOutput:
        raise NotImplementedError()

    @abstractmethod
    def log_prob(self, scores: Any, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def update(self) -> None:
        """ Call to update internal parameters (e.g. decay epsilon) """
        pass


# From preprocessed observation, produce scores for sampler to select action
Scorer = Callable[[Any], Any]

# Transform ReplayBuffer's transition batch to trainer.train
TrainerPreprocessor = Callable[[Any], rlt.PreprocessedTrainingBatch]


""" Called after env.step(action)
Args: (state, actor_output, reward, terminal, possible_actions_mask)
"""
PostStep = Callable[[Any, rlt.ActorOutput, float, bool, Optional[torch.Tensor]], None]


@dataclass
class GaussianSamplerScore(rlt.BaseDataClass):
    loc: torch.Tensor
    scale_log: torch.Tensor
