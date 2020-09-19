#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
import math
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
    reward_clip: float = 1e6
    clip_param: float = 1e6
    normalize: bool = True
    subtract_mean: bool = True
    offset_clamp_min: bool = False
    update_freq: int = 1


class Reinforce(Trainer):
    def __init__(self, actor, params: ReinforceParams):
        self.scorer = actor.scorer
        self.sampler = actor.sampler
        self.params = params
        self.optimizer = params.optimizer.make_optimizer(self.scorer.parameters())
        self.step = 1
        self.losses = []

    def update_model(self):
        if len(self.losses) > 0:
            self.optimizer.zero_grad()
            loss = torch.stack(self.losses).mean()
            loss.backward()
            del self.losses[:]
            self.optimizer.step()

    def train(self, training_batch: rlt.PolicyGradientInput) -> None:
        actions = training_batch.action
        rewards = training_batch.reward.detach()
        scores = self.scorer(training_batch.state)
        characteristic_eligibility = self.sampler.log_prob(scores, actions).float()
        offset_reinforcement = discounted_returns(
            torch.clamp(rewards, max=self.params.reward_clip).clone(), self.params.gamma
        )
        if self.params.normalize:
            offset_reinforcement = whiten(
                offset_reinforcement, subtract_mean=self.params.subtract_mean
            )
        if self.params.offset_clamp_min:
            offset_reinforcement = offset_reinforcement.clamp(min=0)  # pyre-ignore
        if self.params.off_policy:
            target_propensity = self.sampler.log_prob(scores, actions).float()
            characteristic_eligibility = torch.exp(
                torch.clamp(
                    target_propensity - training_batch.log_prob.detach(),
                    max=math.log(float(self.params.clip_param)),
                )
            ).float()
        self.losses.append(-(offset_reinforcement.float()) @ characteristic_eligibility)
        self.step += 1
        if self.step % self.params.update_freq == 0:
            self.update_model()

    def warm_start_components(self) -> List[str]:
        """
        The trainer should specify what members to save and load
        """
        return ["scorer", "actor"]
