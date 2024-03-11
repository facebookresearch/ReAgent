#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
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
        do_log_metrics: bool = False,
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
            if self.normalize or self.subtract_mean:
                raise RuntimeError(
                    "Can't apply a baseline and reward normalization \
                    (or mean subtraction) simultaneously."
                )
            self.value_net = value_net
            self.value_loss_fn = torch.nn.MSELoss(reduction="mean")
        else:
            self.value_net = None
        self.do_log_metrics = do_log_metrics
        if self.do_log_metrics:
            self.losses = []
            self.ips_ratio_means = []

    def _check_input(self, training_batch: rlt.PolicyGradientInput):
        assert training_batch.reward.ndim == 1
        if self.off_policy:
            assert training_batch.log_prob.ndim == 1

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
        self._check_input(training_batch)
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
        elif self.subtract_mean:
            offset_reinforcement -= offset_reinforcement.mean()
        if self.offset_clamp_min:
            offset_reinforcement = offset_reinforcement.clamp(min=0)
        if self.value_net is not None:
            assert not (self.normalize or self.subtract_mean)
            baselines = self.value_net(training_batch.state).squeeze()
            yield self.value_loss_fn(baselines, offset_reinforcement)
            # subtract learned value function baselines from rewards
            offset_reinforcement = offset_reinforcement - baselines

        if self.off_policy:
            characteristic_eligibility = torch.exp(
                torch.clamp(
                    characteristic_eligibility - training_batch.log_prob,
                    max=math.log(float(self.clip_param)),
                )
            ).float()

        loss = -(offset_reinforcement.float().detach()) @ characteristic_eligibility
        if self.do_log_metrics:
            detached_loss = loss.detach().cpu().item() / len(offset_reinforcement)
            self.losses.append(detached_loss)
            detached_ips_ratio_mean = (
                characteristic_eligibility.detach().mean().cpu().item()
            )
            self.ips_ratio_means.append(detached_ips_ratio_mean)
            assert self.logger is not None
            self.logger.log_metrics(
                {
                    "Training_loss/per_iteration": detached_loss,
                    "IPS_ratio_mean/per_iteration": detached_ips_ratio_mean,
                },
                step=self.all_batches_processed,
            )
        yield loss

    def training_epoch_end(self, training_step_outputs):
        if self.do_log_metrics:
            self.logger.log_metrics(
                {
                    "Training_loss/per_epoch": sum(self.losses) / len(self.losses),
                    "IPS_ratio_mean/per_epoch": sum(self.ips_ratio_means)
                    / len(self.ips_ratio_means),
                },
                step=self.current_epoch,
            )
            self.losses = []
            self.ips_ratio_means = []
