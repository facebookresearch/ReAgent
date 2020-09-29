#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import copy
import logging
from typing import List, Optional

import numpy as np
import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.core.configuration import resolve_defaults
from reagent.core.dataclasses import field
from reagent.core.tracker import observable
from reagent.optimizer.union import Optimizer__Union
from reagent.parameters import RLParameters
from reagent.tensorboardX import SummaryWriterContext
from reagent.training.rl_trainer_pytorch import RLTrainer


logger = logging.getLogger(__name__)


@observable(
    td_loss=torch.Tensor,
    reward_loss=torch.Tensor,
    logged_actions=torch.Tensor,
    logged_propensities=torch.Tensor,
    logged_rewards=torch.Tensor,
    model_propensities=torch.Tensor,
    model_rewards=torch.Tensor,
    model_values=torch.Tensor,
    model_action_idxs=torch.Tensor,
)
class SACTrainer(RLTrainer):
    """
    Soft Actor-Critic trainer as described in https://arxiv.org/pdf/1801.01290

    The actor is assumed to implement reparameterization trick.
    """

    @resolve_defaults
    def __init__(
        self,
        actor_network,
        q1_network,
        q2_network=None,
        value_network=None,
        use_gpu: bool = False,
        # Start SACTrainerParameters
        rl: RLParameters = field(default_factory=RLParameters),  # noqa: B008
        q_network_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        value_network_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        actor_network_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        alpha_optimizer: Optional[Optimizer__Union] = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        minibatch_size: int = 1024,
        entropy_temperature: float = 0.01,
        logged_action_uniform_prior: bool = True,
        target_entropy: float = -1.0,
        action_embedding_kld_weight: Optional[float] = None,
        apply_kld_on_mean: bool = False,
        action_embedding_mean: Optional[List[float]] = None,
        action_embedding_variance: Optional[List[float]] = None,
    ) -> None:
        """
        Args:
            actor_network: states -> actions, trained to maximize soft value,
                which is value + policy entropy.
            q1_network: states, action -> q-value
            q2_network (optional): double q-learning to stabilize training
                from overestimation bias
            value_network (optional): states -> value of state under actor
            # alpha in the paper; controlling explore & exploit
            # TODO: finish
        """
        super().__init__(rl, use_gpu=use_gpu)

        self.minibatch_size = minibatch_size
        self.minibatches_per_step = 1

        self.q1_network = q1_network
        self.q1_network_optimizer = q_network_optimizer.make_optimizer(
            q1_network.parameters()
        )

        self.q2_network = q2_network
        if self.q2_network is not None:
            self.q2_network_optimizer = q_network_optimizer.make_optimizer(
                q2_network.parameters()
            )

        self.value_network = value_network
        if self.value_network is not None:
            self.value_network_optimizer = value_network_optimizer.make_optimizer(
                value_network.parameters()
            )
            self.value_network_target = copy.deepcopy(self.value_network)
        else:
            self.q1_network_target = copy.deepcopy(self.q1_network)
            self.q2_network_target = copy.deepcopy(self.q2_network)

        self.actor_network = actor_network
        self.actor_network_optimizer = actor_network_optimizer.make_optimizer(
            actor_network.parameters()
        )
        self.entropy_temperature = entropy_temperature

        self.alpha_optimizer = None
        device = "cuda" if use_gpu else "cpu"
        if alpha_optimizer is not None:
            self.target_entropy = target_entropy
            self.log_alpha = torch.tensor(
                [np.log(self.entropy_temperature)], requires_grad=True, device=device
            )
            self.alpha_optimizer = alpha_optimizer.make_optimizer([self.log_alpha])

        self.logged_action_uniform_prior = logged_action_uniform_prior

        self.add_kld_to_loss = bool(action_embedding_kld_weight)
        self.apply_kld_on_mean = apply_kld_on_mean

        if self.add_kld_to_loss:
            self.kld_weight = action_embedding_kld_weight
            self.action_emb_mean = torch.tensor(action_embedding_mean, device=device)
            self.action_emb_variance = torch.tensor(
                action_embedding_variance, device=device
            )

    def warm_start_components(self):
        components = [
            "q1_network",
            "q1_network_optimizer",
            "actor_network",
            "actor_network_optimizer",
        ]
        if self.q2_network:
            components += ["q2_network", "q2_network_optimizer"]
        if self.value_network:
            components += [
                "value_network",
                "value_network_optimizer",
                "value_network_target",
            ]
        else:
            components += ["q1_network_target"]
            if self.q2_network:
                components += ["q2_network_target"]
        return components

    def train(self, training_batch: rlt.PolicyNetworkInput) -> None:
        """
        IMPORTANT: the input action here is assumed to match the
        range of the output of the actor.
        """

        assert isinstance(training_batch, rlt.PolicyNetworkInput)

        self.minibatch += 1

        state = training_batch.state
        action = training_batch.action
        reward = training_batch.reward
        discount = torch.full_like(reward, self.gamma)
        not_done_mask = training_batch.not_terminal

        # We need to zero out grad here because gradient from actor update
        # should not be used in Q-network update
        self.actor_network_optimizer.zero_grad()
        self.q1_network_optimizer.zero_grad()
        if self.q2_network is not None:
            self.q2_network_optimizer.zero_grad()
        if self.value_network is not None:
            self.value_network_optimizer.zero_grad()

        #
        # First, optimize Q networks; minimizing MSE between
        # Q(s, a) & r + discount * V'(next_s)
        #

        q1_value = self.q1_network(state, action)
        if self.q2_network:
            q2_value = self.q2_network(state, action)
        actor_output = self.actor_network(state)

        # Optimize Alpha
        if self.alpha_optimizer is not None:
            alpha_loss = -(
                (
                    self.log_alpha
                    * (actor_output.log_prob + self.target_entropy).detach()
                ).mean()
            )
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.entropy_temperature = self.log_alpha.exp()

        with torch.no_grad():
            if self.value_network is not None:
                next_state_value = self.value_network_target(
                    training_batch.next_state.float_features
                )
            else:
                next_state_actor_output = self.actor_network(training_batch.next_state)
                next_state_actor_action = (
                    training_batch.next_state,
                    rlt.FeatureData(next_state_actor_output.action),
                )
                next_state_value = self.q1_network_target(*next_state_actor_action)

                if self.q2_network is not None:
                    target_q2_value = self.q2_network_target(*next_state_actor_action)
                    next_state_value = torch.min(next_state_value, target_q2_value)

                log_prob_a = self.actor_network.get_log_prob(
                    training_batch.next_state, next_state_actor_output.action
                )
                log_prob_a = log_prob_a.clamp(-20.0, 20.0)
                next_state_value -= self.entropy_temperature * log_prob_a

            if self.gamma > 0.0:
                target_q_value = (
                    reward + discount * next_state_value * not_done_mask.float()
                )
            else:
                # This is useful in debugging instability issues
                target_q_value = reward

        q1_loss = F.mse_loss(q1_value, target_q_value)
        q1_loss.backward()
        self._maybe_run_optimizer(self.q1_network_optimizer, self.minibatches_per_step)
        if self.q2_network:
            q2_loss = F.mse_loss(q2_value, target_q_value)
            q2_loss.backward()
            self._maybe_run_optimizer(
                self.q2_network_optimizer, self.minibatches_per_step
            )

        # Second, optimize the actor; minimizing KL-divergence between
        # propensity & softmax of value.  Due to reparameterization trick,
        # it ends up being log_prob(actor_action) - Q(s, actor_action)

        state_actor_action = (state, rlt.FeatureData(actor_output.action))
        q1_actor_value = self.q1_network(*state_actor_action)
        min_q_actor_value = q1_actor_value
        if self.q2_network:
            q2_actor_value = self.q2_network(*state_actor_action)
            min_q_actor_value = torch.min(q1_actor_value, q2_actor_value)

        actor_loss = (
            self.entropy_temperature * actor_output.log_prob - min_q_actor_value
        )
        # Do this in 2 steps so we can log histogram of actor loss
        # pyre-fixme[16]: `float` has no attribute `mean`.
        actor_loss_mean = actor_loss.mean()

        if self.add_kld_to_loss:
            if self.apply_kld_on_mean:
                action_batch_m = torch.mean(actor_output.squashed_mean, axis=0)
                action_batch_v = torch.var(actor_output.squashed_mean, axis=0)
            else:
                action_batch_m = torch.mean(actor_output.action, axis=0)
                action_batch_v = torch.var(actor_output.action, axis=0)
            kld = (
                0.5
                * (
                    (action_batch_v + (action_batch_m - self.action_emb_mean) ** 2)
                    / self.action_emb_variance
                    - 1
                    + self.action_emb_variance.log()
                    - action_batch_v.log()
                ).sum()
            )

            actor_loss_mean += self.kld_weight * kld

        actor_loss_mean.backward()
        self._maybe_run_optimizer(
            self.actor_network_optimizer, self.minibatches_per_step
        )

        #
        # Lastly, if applicable, optimize value network; minimizing MSE between
        # V(s) & E_a~pi(s) [ Q(s,a) - log(pi(a|s)) ]
        #

        if self.value_network is not None:
            state_value = self.value_network(state.float_features)

            if self.logged_action_uniform_prior:
                log_prob_a = torch.zeros_like(min_q_actor_value)
                target_value = min_q_actor_value
            else:
                with torch.no_grad():
                    log_prob_a = actor_output.log_prob
                    log_prob_a = log_prob_a.clamp(-20.0, 20.0)
                    target_value = (
                        min_q_actor_value - self.entropy_temperature * log_prob_a
                    )

            value_loss = F.mse_loss(state_value, target_value.detach())
            value_loss.backward()
            self._maybe_run_optimizer(
                self.value_network_optimizer, self.minibatches_per_step
            )

        # Use the soft update rule to update the target networks
        if self.value_network is not None:
            self._maybe_soft_update(
                self.value_network,
                self.value_network_target,
                self.tau,
                self.minibatches_per_step,
            )
        else:
            self._maybe_soft_update(
                self.q1_network,
                self.q1_network_target,
                self.tau,
                self.minibatches_per_step,
            )
            if self.q2_network is not None:
                self._maybe_soft_update(
                    self.q2_network,
                    self.q2_network_target,
                    self.tau,
                    self.minibatches_per_step,
                )

        # Logging at the end to schedule all the cuda operations first
        if (
            self.tensorboard_logging_freq != 0
            and self.minibatch % self.tensorboard_logging_freq == 0
        ):
            SummaryWriterContext.add_histogram("q1/logged_state_value", q1_value)
            if self.q2_network:
                SummaryWriterContext.add_histogram("q2/logged_state_value", q2_value)

            # pyre-fixme[16]: `SummaryWriterContext` has no attribute `add_scalar`.
            SummaryWriterContext.add_scalar(
                "entropy_temperature", self.entropy_temperature
            )
            SummaryWriterContext.add_histogram("log_prob_a", log_prob_a)
            if self.value_network:
                SummaryWriterContext.add_histogram("value_network/target", target_value)

            SummaryWriterContext.add_histogram(
                "q_network/next_state_value", next_state_value
            )
            SummaryWriterContext.add_histogram(
                "q_network/target_q_value", target_q_value
            )
            SummaryWriterContext.add_histogram(
                "actor/min_q_actor_value", min_q_actor_value
            )
            SummaryWriterContext.add_histogram(
                "actor/action_log_prob", actor_output.log_prob
            )
            SummaryWriterContext.add_histogram("actor/loss", actor_loss)
            if self.add_kld_to_loss:
                SummaryWriterContext.add_histogram("kld/mean", action_batch_m)
                SummaryWriterContext.add_histogram("kld/var", action_batch_v)
                SummaryWriterContext.add_scalar("kld/kld", kld)

        self.loss_reporter.report(
            td_loss=float(q1_loss),
            reward_loss=None,
            logged_rewards=reward,
            model_values_on_logged_actions=q1_value,
            model_propensities=actor_output.log_prob.exp(),
            model_values=min_q_actor_value,
        )
