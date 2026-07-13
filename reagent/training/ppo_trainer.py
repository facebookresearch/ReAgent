#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
import inspect
import logging
from dataclasses import field
from typing import Dict, List, Optional, Union

import reagent.core.types as rlt
import torch
import torch.optim
from reagent.core.configuration import resolve_defaults
from reagent.gym.policies.policy import Policy
from reagent.models.base import ModelBase
from reagent.optimizer.union import Optimizer__Union
from reagent.training.reagent_lightning_module import ReAgentLightningModule
from reagent.training.utils import discounted_returns, whiten


logger = logging.getLogger(__name__)

# Upper bound on the per-step importance ratio, applied only to the diagnostic
# ips_value metric to keep the off-policy value estimate from exploding.
IPS_RATIO_CLAMP = 10.0


class PPOTrainer(ReAgentLightningModule):
    """
    Proximal Policy Optimization (PPO). See https://arxiv.org/pdf/1707.06347.pdf
    This is the "clip" version of PPO. It does not include:
    - KL divergence
    Optionally, a value network can be trained and used to form the advantage,
    either as a baseline subtracted from the discounted return
    (reward-to-go - V(s)) or as the one-step TD error
    (r + gamma * V(s') - V(s)) when ``td_error_advantage`` is set.
    The reward-to-go baseline relies on full trajectories up to the terminal
    state being fed in, and trains the value net toward the Monte-Carlo return.
    The TD-error advantage instead trains the value net toward the bootstrapped
    TD(0) target and supports truncated / non-terminal trajectories: when the
    input carries ``next_state`` and ``not_terminal``, the final transition
    bootstraps from V(s') instead of assuming V(s_T) = 0.
    """

    @resolve_defaults
    def __init__(
        self,
        policy: Policy,
        gamma: float = 0.9,
        optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        optimizer_value_net: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        actions: List[str] = field(default_factory=list),  # noqa: B008
        reward_clip: float = 1e6,  # rewards are clamped to this UPPER bound only (no lower bound)
        normalize: bool = True,
        subtract_mean: bool = True,
        offset_clamp_min: bool = False,
        update_freq: int = 1,  # how many env steps between updates
        update_epochs: int = 1,  # how many epochs to run when updating (for PPO)
        ppo_batch_size: int = 1,  # batch size (number of trajectories) used for PPO updates
        ppo_epsilon: float = 0.2,  # clamp importance weights between 1-epsilon and 1+epsilon
        entropy_weight: float = 0.0,  # weight of the entropy term in the PPO loss
        value_net: Optional[ModelBase] = None,
        td_error_advantage: bool = False,  # use one-step TD error (r + gamma*V(s') - V(s)) as the advantage instead of reward-to-go minus baseline
    ):
        # PPO relies on customized update schemas, achieved by manual_backward()
        super().__init__(automatic_optimization=False)
        self.scorer = policy.scorer
        self.sampler = policy.sampler
        self.gamma = gamma
        self.optimizer_value_net = optimizer_value_net
        self.actions = actions
        self.reward_clip = reward_clip
        self.normalize = normalize
        self.subtract_mean = subtract_mean
        self.offset_clamp_min = offset_clamp_min
        self.update_freq = update_freq
        self.update_epochs = update_epochs
        self.ppo_batch_size = ppo_batch_size
        self.ppo_epsilon = ppo_epsilon
        self.entropy_weight = entropy_weight

        self.optimizer = optimizer
        self.value_net = value_net
        self.td_error_advantage = td_error_advantage
        if value_net is not None:
            # reduction="sum" so the critic loss scales with trajectory length,
            # matching the summed PPO surrogate. This intentionally weights longer
            # trajectories more in the value update; switch to "mean" (and reduce
            # later) if per-step weighting is desired.
            self.value_loss_fn = torch.nn.MSELoss(reduction="sum")
            assert not self.normalize, (
                "Can't apply a value baseline and normalize rewards simultaneously"
            )
        if td_error_advantage:
            assert value_net is not None, (
                "td_error_advantage requires a value_net to estimate V(s)"
            )
        assert (ppo_epsilon >= 0) and (ppo_epsilon <= 1), (
            "ppo_epsilon has to be in [0;1]"
        )
        assert update_freq >= 1, "update_freq has to be >= 1"
        assert update_epochs >= 1, "update_epochs has to be >= 1"
        assert ppo_batch_size >= 1, "ppo_batch_size has to be >= 1"

        self.traj_buffer = []

    def _trajectory_to_losses(
        self, trajectory: rlt.PolicyGradientInput
    ) -> Dict[str, torch.Tensor]:
        """
        Get a dict of losses for the trajectory. Dict always includes PPO loss.
        If a value baseline is trained, a loss for the value network is also included.
        """
        self._check_input(trajectory)
        losses = {}
        actions = trajectory.action
        rewards = trajectory.reward.detach()
        scores = self.scorer(*self._scorer_inputs(trajectory))

        # Detached per-step advantage weights for the surrogate; also populates
        # losses["value_net_loss"] when a value net is trained.
        offset_reinforcement = self._compute_advantage(trajectory, rewards, losses)

        target_propensity = self.sampler.log_prob(scores, actions).float()
        characteristic_eligibility = torch.exp(
            target_propensity - trajectory.log_prob.detach()
        ).float()

        # Per-timestep PPO clip, then sum over the trajectory. The min must be
        # taken element-wise BEFORE summing; min(sum, sum) would apply the clip
        # at the trajectory level and change the gradient.
        offset_reinforcement = offset_reinforcement.float()
        surrogate = torch.min(
            offset_reinforcement * characteristic_eligibility,
            offset_reinforcement
            * torch.clamp(
                characteristic_eligibility,
                1 - self.ppo_epsilon,
                1 + self.ppo_epsilon,
            ),
        )
        losses["ppo_loss"] = -surrogate.sum()
        if self.entropy_weight != 0:
            # sampler.entropy returns the per-step mean; scale by the number of
            # steps so the entropy bonus is summed over the trajectory to match
            # the summed surrogate (otherwise its weight scales as 1 / len).
            entropy = self.sampler.entropy(scores) * scores.shape[0]
            # "-" bcs minimizing, not maximizing
            losses["ppo_loss"] = losses["ppo_loss"] - self.entropy_weight * entropy
        return losses

    def _check_input(self, trajectory: rlt.PolicyGradientInput) -> None:
        assert trajectory.action.ndim == 2, (
            f"action must be 2-D, got {trajectory.action.shape}"
        )
        trajectory_length = trajectory.action.shape[0]
        assert trajectory_length > 0, "trajectory must contain at least one step"
        assert trajectory.reward.ndim == 1, (
            f"reward must be 1-D, got {trajectory.reward.shape}"
        )
        assert trajectory.log_prob.ndim == 1, (
            f"log_prob must be 1-D, got {trajectory.log_prob.shape}"
        )
        assert trajectory.reward.shape[0] == trajectory_length, (
            f"reward length {trajectory.reward.shape[0]} != action length {trajectory_length}"
        )
        assert trajectory.log_prob.shape[0] == trajectory_length, (
            f"log_prob length {trajectory.log_prob.shape[0]} != action length {trajectory_length}"
        )
        if trajectory.possible_actions_mask is not None:
            assert trajectory.possible_actions_mask.ndim == 2, (
                "possible_actions_mask must be 2-D, "
                f"got {trajectory.possible_actions_mask.shape}"
            )
            assert trajectory.possible_actions_mask.shape[0] == trajectory_length, (
                "possible_actions_mask length "
                f"{trajectory.possible_actions_mask.shape[0]} != action length {trajectory_length}"
            )
        if trajectory.not_terminal is not None:
            assert trajectory.not_terminal.ndim == 1, (
                f"not_terminal must be 1-D, got {trajectory.not_terminal.shape}"
            )
            assert trajectory.not_terminal.shape[0] == trajectory_length, (
                f"not_terminal length {trajectory.not_terminal.shape[0]} != action length {trajectory_length}"
            )
        if trajectory.next_state is not None:
            assert trajectory.next_state.float_features.shape[0] == trajectory_length, (
                f"next_state length {trajectory.next_state.float_features.shape[0]} "
                f"!= action length {trajectory_length}"
            )

    def _scorer_inputs(self, trajectory: rlt.PolicyGradientInput) -> List:
        """Build the positional inputs for the scorer, shared by the loss and
        eval-metric paths so they never diverge (e.g. on the GNN graph path)."""
        scorer_inputs = []
        if inspect.getattr_static(trajectory, "graph", None) is not None:
            # TODO: can this line be hit currently in ReAgent?
            # GNN
            scorer_inputs.append(trajectory.graph)
        else:
            scorer_inputs.append(trajectory.state)
        if trajectory.possible_actions_mask is not None:
            scorer_inputs.append(trajectory.possible_actions_mask)
        return scorer_inputs

    def _compute_advantage(
        self,
        trajectory: rlt.PolicyGradientInput,
        rewards: torch.Tensor,
        losses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return the detached per-step advantage used to weight the surrogate.

        With ``td_error_advantage`` set, delegates to the bootstrapped one-step
        TD error. Otherwise uses the discounted reward-to-go, optionally whitened
        and/or with a learned value baseline subtracted. Populates
        ``losses["value_net_loss"]`` whenever a value net is trained.
        """
        if self.value_net is not None and self.td_error_advantage:
            return self._td_error_advantage(trajectory, rewards, losses)

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
            # reshape(-1) guards length-1 trajectories, whose squeeze() is 0-dim.
            baselines = self.value_net(trajectory.state).squeeze().reshape(-1)
            # use reward-to-go as label for training the value function
            losses["value_net_loss"] = self.value_loss_fn(
                baselines, offset_reinforcement
            )
            # detach bcs we want PPO to tweak policy, not baseline
            offset_reinforcement = offset_reinforcement - baselines.detach()
        return offset_reinforcement

    def _td_error_advantage(
        self,
        trajectory: rlt.PolicyGradientInput,
        rewards: torch.Tensor,
        losses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """One-step TD error advantage.

        A_t = r_t + gamma * not_terminal_t * V(s_{t+1}) - V(s_t).

        The value net is trained toward the same (detached) bootstrapped target,
        so the critic stays consistent with the advantage and unbiased for
        truncated trajectories. reshape(-1) keeps length-1 trajectories 1-D.
        """
        baselines = self.value_net(trajectory.state).reshape(-1)  # V(s_t)
        values = baselines.detach()
        clamped_rewards = torch.clamp(rewards, max=self.reward_clip).reshape(-1)
        not_terminal = self._not_terminal_mask(trajectory, values)

        if trajectory.next_state is not None:
            # Bootstrap from each actual next state; supports truncated
            # (non-terminal) final transitions.
            next_values = self.value_net(trajectory.next_state).detach().reshape(-1)
        else:
            # Without next states we can only bootstrap interior steps from the
            # next row in the trajectory; the final transition must be terminal.
            self._assert_final_step_terminal(trajectory, not_terminal)
            next_values = torch.cat([values[1:], values.new_zeros(1)])
        # Semi-gradient TD(0) target (detached); the mask drops the bootstrap for
        # terminal transitions and keeps it for truncated ones.
        td_target = clamped_rewards + self.gamma * not_terminal * next_values
        losses["value_net_loss"] = self.value_loss_fn(baselines, td_target)
        advantage = td_target - values
        if self.offset_clamp_min:
            # Honor offset_clamp_min for the TD advantage as well. (The reward-to-go
            # path clamps the return before the baseline; here we clamp the final
            # advantage, since the TD error is itself the advantage.)
            advantage = advantage.clamp(min=0)
        return advantage

    def _assert_final_step_terminal(
        self, trajectory: rlt.PolicyGradientInput, not_terminal: torch.Tensor
    ) -> None:
        """Guard the no-next_state bootstrap paths: without next_state the final
        transition cannot bootstrap V(s_T), so it must be terminal. A missing
        not_terminal already zeroes the last step (default mask), so only the
        explicit-mask case needs the (host-syncing) scalar read.
        """
        assert trajectory.not_terminal is None or bool(not_terminal[-1] == 0), (
            "a truncated final transition (not_terminal[-1] != 0) needs next_state "
            "(and value_net) to bootstrap V(s_T); pass next_state or mark the last "
            "step terminal"
        )

    def _not_terminal_mask(
        self, trajectory: rlt.PolicyGradientInput, reference: torch.Tensor
    ) -> torch.Tensor:
        if trajectory.not_terminal is not None:
            return trajectory.not_terminal.detach().reshape(-1).to(reference)
        # Per the PolicyGradientInput contract, a missing not_terminal means a
        # complete episode ending at a terminal state: bootstrap the interior
        # steps but never the final one (even if next_state is supplied).
        not_terminal = torch.ones_like(reference)
        not_terminal[-1] = 0.0
        return not_terminal

    def _trajectory_returns(self, trajectory: rlt.PolicyGradientInput) -> torch.Tensor:
        rewards = torch.clamp(trajectory.reward.detach(), max=self.reward_clip)
        returns = torch.empty_like(rewards, dtype=torch.float)
        not_terminal = self._not_terminal_mask(trajectory, rewards)
        if trajectory.next_state is not None and self.value_net is not None:
            # V(s_T) bootstrap; the loop below applies not_terminal[-1] exactly
            # once, so it must NOT be pre-multiplied here.
            next_values = self.value_net(trajectory.next_state).detach().reshape(-1)
            running = next_values[-1]
        else:
            self._assert_final_step_terminal(trajectory, not_terminal)
            running = torch.zeros((), dtype=torch.float, device=rewards.device)

        for t in range(rewards.shape[0] - 1, -1, -1):
            running = rewards[t].float() + self.gamma * not_terminal[t] * running
            returns[t] = running
        return returns

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

    def get_optimizers(self):
        opts = self.optimizers()
        if self.value_net is not None:
            return opts[0], opts[1]
        return None, opts[0]

    # pyre-fixme[14]: `training_step` overrides method defined in
    #  `ReAgentLightningModule` inconsistently.
    def training_step(
        self,
        training_batch: Union[rlt.PolicyGradientInput, Dict[str, torch.Tensor]],
        batch_idx: int,
    ):
        if isinstance(training_batch, dict):
            training_batch = rlt.PolicyGradientInput.from_dict(training_batch)

        self.traj_buffer.append(training_batch)
        if len(self.traj_buffer) == self.update_freq:
            self.update_model()

    def update_model(self):
        assert len(self.traj_buffer) == self.update_freq, (
            "trajectory buffer does not have sufficient samples for model_update"
        )
        for _ in range(self.update_epochs):
            # iterate through minibatches of PPO updates in random order
            random_order = torch.randperm(len(self.traj_buffer))
            for i in range(0, len(self.traj_buffer), self.ppo_batch_size):
                idx = random_order[i : i + self.ppo_batch_size]
                training_batch_list = [self.traj_buffer[i] for i in idx]
                self._update_model(training_batch_list)

        self.traj_buffer = []  # empty the buffer

    def _update_model(self, training_batch_list: List[rlt.PolicyGradientInput]):
        losses = {
            "ppo_loss": [],
            "value_net_loss": [],
        }
        value_net_opt, ppo_opt = self.get_optimizers()

        for traj in training_batch_list:
            loss = self._trajectory_to_losses(traj)
            for k, v in loss.items():
                losses[k].append(v)

        if self.value_net is not None:
            # TD loss for the baseline value network
            value_net_loss = torch.stack(losses["value_net_loss"]).sum()
            value_net_opt.zero_grad()
            self.manual_backward(value_net_loss)
            value_net_opt.step()

        # PPO "loss" for the policy network
        ppo_loss = torch.stack(losses["ppo_loss"]).sum()
        ppo_opt.zero_grad()
        self.manual_backward(ppo_loss)
        ppo_opt.step()

        # Report training metrics so they surface in the training output.
        self.reporter.log(
            ppo_loss=ppo_loss.detach().reshape(1),
            value_net_loss=(
                value_net_loss.detach().reshape(1)
                if self.value_net is not None
                else torch.zeros(1)
            ),
        )
        # Populate logger_data (the flow's line-plot metrics summary). Loggers
        # expect Python floats; gather every metric as a 0-dim tensor and do a
        # single device->host transfer instead of one .item() sync per metric.
        if self.logger is not None:
            metric_tensors = {"Training_loss/ppo_loss": ppo_loss.detach()}
            if self.value_net is not None:
                metric_tensors["Training_loss/value_net_loss"] = value_net_loss.detach()
            metric_tensors.update(self._eval_metrics(training_batch_list))
            keys = list(metric_tensors)
            values = (
                torch.stack([metric_tensors[k].reshape(()) for k in keys])
                .cpu()
                .tolist()
            )
            self.logger.log_metrics(
                dict(zip(keys, values)), step=self.all_batches_processed
            )

    @torch.no_grad()
    def _eval_metrics(
        self, training_batch_list: List[rlt.PolicyGradientInput]
    ) -> Dict[str, torch.Tensor]:
        """Interpretable, policy-relevant signals (unlike the PPO surrogate loss).

        - mean_reward: average logged reward across the update's trajectories
          (a scale/sanity check on the data).
        - ips_value: a one-step importance-sampled off-policy value estimate
          E[(pi/mu) * return]. It rises as the policy shifts probability toward
          higher-return actions, so it tracks policy quality. The importance
          ratio is clamped to keep the estimate from exploding.
        """
        rewards = torch.cat([traj.reward.reshape(-1) for traj in training_batch_list])
        ips_values = []
        for traj in training_batch_list:
            self._check_input(traj)
            scores = self.scorer(*self._scorer_inputs(traj))
            log_prob = self.sampler.log_prob(scores, traj.action).float()
            ratio = torch.exp(log_prob - traj.log_prob).clamp(max=IPS_RATIO_CLAMP)
            returns = self._trajectory_returns(traj)
            ips_values.append((ratio * returns).mean())
        return {
            "Training/mean_reward": rewards.mean().detach(),
            "Training/ips_value": torch.stack(ips_values).mean().detach(),
        }
