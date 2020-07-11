#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from dataclasses import dataclass, field
from typing import List

import reagent.types as rlt
import torch
import torch.optim
from reagent.optimizer.union import Optimizer__Union
from reagent.training.trainer import Trainer
from reagent.training.utils import discounted_returns, whiten


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReinforceParams:
    gamma: float = 0.0
    optimizer: Optimizer__Union = field(default_factory=Optimizer__Union.default)
    off_policy: bool = False
    clip_param: float = 1e6
    normalize: bool = True
    subtract_mean: bool = True
    offset_clamp_min: bool = True


class Reinforce(Trainer):
    def __init__(self, actor, params: ReinforceParams):
        self.scorer = actor.scorer
        self.sampler = actor.sampler
        self.params = params
        self.optimizer = params.optimizer.make_optimizer(self.scorer.parameters())

    def train(self, training_batch: rlt.PolicyGradientInput) -> None:
        actions = training_batch.action
        rewards = training_batch.reward.detach()
        scores = self.scorer(training_batch.state)
        characteristic_eligibility = self.sampler.log_prob(scores, actions).float()
        offset_reinforcement = discounted_returns(rewards, self.params.gamma)
        if self.params.normalize:
            offset_reinforcement = whiten(
                offset_reinforcement, subtract_mean=self.params.subtract_mean
            )
        if self.params.offset_clamp_min:
            offset_reinforcement = offset_reinforcement.clamp(min=0)  # pyre-ignore
        correction = 1.0
        if self.params.off_policy:
            correction = torch.exp(characteristic_eligibility - training_batch.log_prob)
            correction *= (correction < self.params.clip_param).float()
            characteristic_eligibility *= correction.detach()
        err = -(offset_reinforcement.float()) @ characteristic_eligibility
        self.optimizer.zero_grad()
        err.backward()
        self.optimizer.step()

    def warm_start_components(self) -> List[str]:
        """
        The trainer should specify what members to save and load
        """
        return ["scorer", "actor"]
