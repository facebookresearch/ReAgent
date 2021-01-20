#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from dataclasses import field
from typing import Dict, List, Optional

import reagent.types as rlt
import torch
import torch.optim
from reagent.core.configuration import resolve_defaults
from reagent.gym.policies.policy import Policy
from reagent.models.base import ModelBase
from reagent.optimizer.union import Optimizer__Union
from reagent.training.trainer import Trainer
from reagent.training.utils import discounted_returns, whiten


logger = logging.getLogger(__name__)


class PPOTrainer(Trainer):
    """
    Proximal Policy Optimization (PPO). See https://arxiv.org/pdf/1707.06347.pdf
    This is the "clip" version of PPO. It does not include:
    - KL divergence
    - Bootstrapping with a critic model (this only works if full trajectories up to terminal state are fed in)
    Optionally, a value network can be trained and used as a baseline for rewards.
    """

    @resolve_defaults
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
        off_policy: bool = False,
        reward_clip: float = 1e6,
        normalize: bool = True,
        subtract_mean: bool = True,
        offset_clamp_min: bool = False,
        update_freq: int = 100,  # how many env steps between updates
        update_epochs: int = 5,  # how many epochs to run when updating (for PPO)
        ppo_batch_size: int = 10,  # batch size (number of trajectories) used for PPO updates
        ppo_epsilon: float = 0.2,  # clamp importance weights between 1-epsilon and 1+epsilon
        entropy_weight: float = 0.0,  # weight of the entropy term in the PPO loss
        value_net: Optional[ModelBase] = None,
    ):
        self.scorer = policy.scorer
        self.sampler = policy.sampler
        self.gamma = gamma
        self.optimizer_value_net = optimizer_value_net
        self.off_policy = off_policy
        self.reward_clip = reward_clip
        self.normalize = normalize
        self.subtract_mean = subtract_mean
        self.offset_clamp_min = offset_clamp_min
        self.update_freq = update_freq
        self.update_epochs = update_epochs
        self.ppo_batch_size = ppo_batch_size
        self.ppo_epsilon = ppo_epsilon
        self.entropy_weight = entropy_weight

        self.optimizer = optimizer.make_optimizer(self.scorer.parameters())
        if value_net is not None:
            self.value_net = value_net
            self.value_net_optimizer = optimizer_value_net.make_optimizer(
                self.value_net.parameters()
            )
            self.value_loss_fn = torch.nn.MSELoss(reduction="mean")
        else:
            self.value_net = None
            self.value_net_optimizer = None
        assert (ppo_epsilon >= 0) and (
            ppo_epsilon <= 1
        ), "ppo_epslion has to be in [0;1]"
        self.step = 0
        self.traj_buffer = []

    def update_model(self):
        """
        Iterate through the PPO trajectory buffer `update_epochs` times, sampling minibatches
        of `ppo_batch_size` trajectories. Perform gradient ascent on the clipped PPO loss.
        If value network is being trained, also perform gradient descent steps for its loss.
        """
        assert len(self.traj_buffer) == self.update_freq
        for _ in range(self.update_epochs):
            # iterate through minibatches of PPO updates in random order
            random_order = torch.randperm(len(self.traj_buffer))
            for i in range(0, len(self.traj_buffer), self.ppo_batch_size):
                idx = random_order[i : i + self.ppo_batch_size]
                # get the losses for the sampled trajectories
                ppo_loss = []
                value_net_loss = []
                for i in idx:
                    traj_losses = self._trajectory_to_losses(self.traj_buffer[i])
                    ppo_loss.append(traj_losses["ppo_loss"])
                    if self.value_net_optimizer is not None:
                        value_net_loss.append(traj_losses["value_net_loss"])
                self.optimizer.zero_grad()
                ppo_loss = torch.stack(ppo_loss).mean()
                ppo_loss.backward()
                self.optimizer.step()
                if self.value_net_optimizer is not None:
                    self.value_net_optimizer.zero_grad()
                    value_net_loss = torch.stack(value_net_loss).mean()
                    value_net_loss.backward()
                    self.value_net_optimizer.step()
        self.traj_buffer = []  # empty the buffer

    def train(self, training_batch: rlt.PolicyGradientInput) -> None:
        self.traj_buffer.append(training_batch)
        self.step += 1
        if self.step % self.update_freq == 0:
            self.update_model()

    def _trajectory_to_losses(
        self, trajectory: rlt.PolicyGradientInput
    ) -> Dict[str, torch.Tensor]:
        """
        Get a dict of losses for the trajectory. Dict always includes PPO loss.
        If a value baseline is trained, a loss for the value network is also included.
        """
        losses = {}
        actions = trajectory.action
        rewards = trajectory.reward.detach()
        scores = self.scorer(trajectory.state, trajectory.possible_actions_mask)
        offset_reinforcement = discounted_returns(
            torch.clamp(rewards, max=self.reward_clip).clone(), self.gamma
        )
        if self.normalize:
            offset_reinforcement = whiten(
                offset_reinforcement, subtract_mean=self.subtract_mean
            )
        if self.offset_clamp_min:
            offset_reinforcement = offset_reinforcement.clamp(min=0)  # pyre-ignore
        if self.value_net is not None:
            if self.normalize:
                raise RuntimeError(
                    "Can't apply a baseline and normalize rewards simultaneously"
                )
            # subtract learned value function baselines from rewards
            baselines = self.value_net(trajectory.state).squeeze()
            # use reward-to-go as label for training the value function
            losses["value_net_loss"] = self.value_loss_fn(
                baselines, offset_reinforcement
            )
            # detach bcs we want PPO to tweak policy, not baseline
            offset_reinforcement = offset_reinforcement - baselines.detach()

        target_propensity = self.sampler.log_prob(scores, actions).float()
        characteristic_eligibility = torch.exp(
            target_propensity - trajectory.log_prob.detach()
        ).float()

        losses["ppo_loss"] = -torch.min(
            offset_reinforcement.float() @ characteristic_eligibility,
            offset_reinforcement.float()
            @ torch.clamp(
                characteristic_eligibility,
                1 - self.ppo_epsilon,
                1 + self.ppo_epsilon,
            ),
        )
        if self.entropy_weight != 0:
            entropy = self.sampler.entropy(scores)
            # "-" bcs minimizing, not maximizing
            losses["ppo_loss"] = losses["ppo_loss"] - self.entropy_weight * entropy
        return losses

    def warm_start_components(self) -> List[str]:
        """
        The trainer should specify what members to save and load
        """
        return ["scorer", "policy"]
