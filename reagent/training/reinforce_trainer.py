#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import inspect
import logging
import math
from dataclasses import field
from typing import List, Optional

import reagent.core.types as rlt
import torch
import torch.optim
from reagent.gym.policies.policy import Policy
from reagent.models.base import ModelBase
from reagent.optimizer.union import Optimizer__Union
from reagent.training.reagent_lightning_module import ReAgentLightningModule
from reagent.training.utils import discounted_returns, whiten

logger = logging.getLogger(__name__)


class ReinforceTrainer(ReAgentLightningModule):
    def __init__(
        self,
        policy: Policy,
        gamma: float = 0.0,
        optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        optimizer_value_net: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        actions: List[str] = field(default_factory=list),  # noqa: B008
        off_policy: bool = False,
        reward_clip: float = 1e6,
        clip_param: float = 1e6,
        normalize: bool = True,
        subtract_mean: bool = True,
        offset_clamp_min: bool = False,
        value_net: Optional[ModelBase] = None,
    ):
        super().__init__()
        self._actions = actions
        self.scorer = policy.scorer
        self.sampler = policy.sampler
        self.gamma = gamma
        self.off_policy = off_policy
        self.reward_clip = reward_clip
        self.clip_param = clip_param
        self.normalize = normalize
        self.subtract_mean = subtract_mean
        self.offset_clamp_min = offset_clamp_min
        self.optimizer = optimizer
        self.optimizer_value_net = optimizer_value_net
        if value_net is not None:
            self.value_net = value_net
            self.value_loss_fn = torch.nn.MSELoss(reduction="mean")
        else:
            self.value_net = None

    def configure_optimizers(self):
        optimizers = []
        # value net optimizer
        if self.value_net is not None:
            optimizers.append(
                self.optimizer_value_net.make_optimizer_scheduler(
                    self.value_net.parameters()
                )
            )
        # policy optimizer
        optimizers.append(
            self.optimizer.make_optimizer_scheduler(self.scorer.parameters())
        )

        return optimizers

    def train_step_gen(self, training_batch: rlt.PolicyGradientInput, batch_idx: int):
        actions = training_batch.action
        rewards = training_batch.reward.detach()
        scorer_inputs = []
        if inspect.getattr_static(training_batch, "graph", None) is not None:
            # GNN
            scorer_inputs.append(training_batch.graph)
        else:
            scorer_inputs.append(training_batch.state)
        if training_batch.possible_actions_mask is not None:
            scorer_inputs.append(training_batch.possible_actions_mask)
        scores = self.scorer(*scorer_inputs)
        characteristic_eligibility = self.sampler.log_prob(scores, actions).float()
        offset_reinforcement = discounted_returns(
            torch.clamp(rewards, max=self.reward_clip).clone(), self.gamma
        )
        if self.normalize:
            offset_reinforcement = whiten(
                offset_reinforcement, subtract_mean=self.subtract_mean
            )
        if self.offset_clamp_min:
            offset_reinforcement = offset_reinforcement.clamp(min=0)
        if self.value_net is not None:
            if self.normalize:
                raise RuntimeError(
                    "Can't apply a baseline and normalize rewards simultaneously"
                )
            baselines = self.value_net(training_batch.state).squeeze()
            yield self.value_loss_fn(baselines, offset_reinforcement)
            # subtract learned value function baselines from rewards
            offset_reinforcement = offset_reinforcement - baselines

        if self.off_policy:
            target_propensity = self.sampler.log_prob(scores, actions).float()
            characteristic_eligibility = torch.exp(
                torch.clamp(
                    target_propensity - training_batch.log_prob,
                    max=math.log(float(self.clip_param)),
                )
            ).float()
        yield -(offset_reinforcement.float()) @ characteristic_eligibility  # PG "loss"
