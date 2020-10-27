#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import reagent.types as rlt
import torch
import torch.optim
from reagent.models.base import ModelBase
from reagent.optimizer.union import Optimizer__Union
from reagent.training.trainer import Trainer
from reagent.training.utils import discounted_returns, whiten


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReinforceParams:
    gamma: float = 0.0
    optimizer: Optimizer__Union = field(default_factory=Optimizer__Union.default)
    optimizer_value_net: Optimizer__Union = field(
        default_factory=Optimizer__Union.default
    )
    off_policy: bool = False
    reward_clip: float = 1e6
    clip_param: float = 1e6
    normalize: bool = True
    subtract_mean: bool = True
    offset_clamp_min: bool = False
    update_freq: int = 1


class Reinforce(Trainer):
    def __init__(
        self, actor, params: ReinforceParams, value_net: Optional[ModelBase] = None
    ):
        self.scorer = actor.scorer
        self.sampler = actor.sampler
        self.params = params
        self.optimizer = params.optimizer.make_optimizer(self.scorer.parameters())
        if value_net is not None:
            self.value_net = value_net
            self.value_net_optimizer = params.optimizer_value_net.make_optimizer(
                self.value_net.parameters()
            )
            self.value_loss_fn = torch.nn.MSELoss(reduction="mean")
            self.value_net_losses = []
        else:
            self.value_net = None
            self.value_net_optimizer = None
        self.step = 1
        self.losses = []

    def update_model(self):
        if len(self.losses) > 0:
            self.optimizer.zero_grad()
            loss = torch.stack(self.losses).mean()
            loss.backward()
            del self.losses[:]
            self.optimizer.step()
            if self.value_net_optimizer is not None:
                self.value_net_optimizer.zero_grad()
                value_net_loss = torch.stack(self.value_net_losses).mean()
                value_net_loss.backward()
                del self.value_net_losses[:]
                self.value_net_optimizer.step()

    def train(self, training_batch: rlt.PolicyGradientInput) -> None:
        actions = training_batch.action
        rewards = training_batch.reward.detach()
        scores = self.scorer(training_batch.state, training_batch.possible_actions_mask)
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
        if self.value_net is not None:
            if self.params.normalize:
                raise RuntimeError(
                    "Can't apply a baseline and normalize rewards simultaneously"
                )
            # subtract learned value function baselines from rewards
            baselines = self.value_net(training_batch.state).squeeze()
            # use reward-to-go as label for training the value function
            self.value_net_losses.append(
                self.value_loss_fn(baselines, offset_reinforcement)
            )
            # detach bcs we want REINFORCE to tweak policy, not baseline
            offset_reinforcement = offset_reinforcement - baselines.detach()

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
