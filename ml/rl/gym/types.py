#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

import ml.rl.types as rlt
import numpy as np
import torch
from ml.rl.replay_memory.circular_replay_buffer import ReplayBuffer


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

# Transform gym.Env's observation to Scorer's input
PolicyPreprocessor = Callable[[Any], Any]


# Transform sampled action to input to gym.Env.step
ActionPreprocessor = Callable[[rlt.ActorOutput], np.array]


ObservationType = Any
RewardType = float
TerminalType = bool
PossibleActionsMaskType = Optional[torch.Tensor]
ReplayBufferAddFn = Callable[
    [
        ReplayBuffer,
        ObservationType,
        rlt.ActorOutput,
        RewardType,
        TerminalType,
        PossibleActionsMaskType,
    ],
    None,
]

# Called in post_step of Agent to train on sampled batch from RB
ReplayBufferTrainFn = Callable[[ReplayBuffer], None]


@dataclass
class GaussianSamplerScore(rlt.BaseDataClass):
    loc: torch.Tensor
    scale_log: torch.Tensor
