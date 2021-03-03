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
from reagent.optimizer.optimizer import Optimizer
from reagent.optimizer.union import Optimizer__Union
from reagent.training.reagent_lightning_module import ReAgentLightningModule
from reagent.training.utils import discounted_returns, whiten


logger = logging.getLogger(__name__)


class PPOTrainer(ReAgentLightningModule):
    """
    Proximal Policy Optimization (PPO). See https://arxiv.org/pdf/1707.06347.pdf
    This is the "clip" version of PPO. It does not include:
    - KL divergence
    - Bootstrapping with a critic model (our approach only works if full trajectories up to terminal state are fed in)
    Optionally, a value network can be trained and used as a baseline for rewards.
    Note that update frequency, number of epochs and batch size have to be specified in EpisodicDatasetDataloader
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
        actions: List[str] = field(default_factory=list),  # noqa: B008
        reward_clip: float = 1e6,
        normalize: bool = True,
        subtract_mean: bool = True,
        offset_clamp_min: bool = False,
        ppo_epsilon: float = 0.2,  # clamp importance weights between 1-epsilon and 1+epsilon
        entropy_weight: float = 0.0,  # weight of the entropy term in the PPO loss
        value_net: Optional[ModelBase] = None,
    ):
        super().__init__()
        self.scorer = policy.scorer
        self.sampler = policy.sampler
        self.gamma = gamma
        self.optimizer_value_net = optimizer_value_net
        self.actions = actions
        self.reward_clip = reward_clip
        self.normalize = normalize
        self.subtract_mean = subtract_mean
        self.offset_clamp_min = offset_clamp_min
        self.ppo_epsilon = ppo_epsilon
        self.entropy_weight = entropy_weight

        self.optimizer = optimizer
        self.value_net = value_net
        if value_net is not None:
            self.value_loss_fn = torch.nn.MSELoss(reduction="mean")
        assert (ppo_epsilon >= 0) and (
            ppo_epsilon <= 1
        ), "ppo_epslion has to be in [0;1]"

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
            baselines = self.value_net(trajectory.state).squeeze()  # pyre-ignore
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

    def configure_optimizers(self) -> List[Optimizer]:
        optimizers = []
        # value net optimizer
        if self.value_net is not None:
            optimizers.append(
                self.optimizer_value_net.make_optimizer(self.value_net.parameters())  # pyre-ignore
            )
        # policy optimizer
        optimizers.append(self.optimizer.make_optimizer(self.scorer.parameters()))
        return optimizers

    def train_step_gen(
        self, training_batch: List[rlt.PolicyGradientInput], batch_idx: int
    ):
        losses = {
            "ppo_loss": [],
            "value_net_loss": [],
        }
        for traj in training_batch:
            loss = self._trajectory_to_losses(traj)
            for k, v in loss.items():
                losses[k].append(v)
        if self.value_net is not None:
            # TD loss for the baseline value network
            yield torch.stack(losses["value_net_loss"]).sum()
        # PPO "loss" for the policy network
        yield torch.stack(losses["ppo_loss"]).sum()
