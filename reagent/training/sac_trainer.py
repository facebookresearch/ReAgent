#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import copy
import logging
from typing import List, Optional

import numpy as np
import reagent.core.types as rlt
import torch
import torch.nn.functional as F
from reagent.core.configuration import resolve_defaults
from reagent.core.dataclasses import dataclass
from reagent.core.dataclasses import field
from reagent.core.parameters import RLParameters
from reagent.optimizer import Optimizer__Union, SoftUpdate
from reagent.training.reagent_lightning_module import ReAgentLightningModule
from reagent.training.rl_trainer_pytorch import RLTrainerMixin


logger = logging.getLogger(__name__)


@dataclass
class CRRWeightFn:
    # pick indicator or exponent
    indicator_fn_threshold: Optional[float] = None
    exponent_beta: Optional[float] = None
    exponent_clamp: Optional[float] = None

    def __post_init_post_parse__(self):
        assert self.exponent_beta or self.indicator_fn_threshold
        assert not (self.exponent_beta and self.indicator_fn_threshold)
        if self.exponent_beta:
            assert self.exponent_beta > 1e-6

        if self.exponent_clamp:
            assert self.exponent_clamp > 1e-6

    def get_weight_from_advantage(self, advantage):
        if self.indicator_fn_threshold:
            return (advantage >= self.indicator_fn_threshold).float()

        if self.exponent_beta:
            exp = torch.exp(advantage / self.exponent_beta)
            if self.exponent_clamp:
                exp = torch.clamp(exp, 0.0, self.exponent_clamp)
            return exp


class SACTrainer(RLTrainerMixin, ReAgentLightningModule):
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
        crr_config: Optional[CRRWeightFn] = None,
        backprop_through_log_prob: bool = True,
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
            backprop_through_log_prob: This is mostly for backward compatibility issue;
                we used to have a bug that does this and it yields a better result in
                some cases
            # TODO: finish
        """
        super().__init__()
        self.rl_parameters = rl
        self.q1_network = q1_network
        self.q2_network = q2_network
        self.q_network_optimizer = q_network_optimizer

        self.value_network = value_network
        self.value_network_optimizer = value_network_optimizer
        if self.value_network is not None:
            self.value_network_target = copy.deepcopy(self.value_network)
        else:
            self.q1_network_target = copy.deepcopy(self.q1_network)
            self.q2_network_target = copy.deepcopy(self.q2_network)

        self.actor_network = actor_network
        self.actor_network_optimizer = actor_network_optimizer
        self.entropy_temperature = entropy_temperature

        self.alpha_optimizer = alpha_optimizer
        if alpha_optimizer is not None:
            self.target_entropy = target_entropy
            self.log_alpha = torch.nn.Parameter(
                torch.tensor([np.log(self.entropy_temperature)])
            )

        self.logged_action_uniform_prior = logged_action_uniform_prior

        self.add_kld_to_loss = bool(action_embedding_kld_weight)
        self.apply_kld_on_mean = apply_kld_on_mean

        if self.add_kld_to_loss:
            self.kld_weight = action_embedding_kld_weight
            # Calling register_buffer so that the tensors got moved to the right device
            self.register_buffer("action_emb_mean", None)
            self.register_buffer("action_emb_variance", None)
            # Assigning the values here instead of above so that typechecker wouldn't complain
            self.action_emb_mean = torch.tensor(action_embedding_mean)
            self.action_emb_variance = torch.tensor(action_embedding_variance)

        self.crr_config = crr_config
        if crr_config:
            assert self.value_network is not None

        self.backprop_through_log_prob = backprop_through_log_prob

    def configure_optimizers(self):
        optimizers = []

        optimizers.append(
            self.q_network_optimizer.make_optimizer_scheduler(
                self.q1_network.parameters()
            )
        )
        if self.q2_network:
            optimizers.append(
                self.q_network_optimizer.make_optimizer_scheduler(
                    self.q2_network.parameters()
                )
            )
        optimizers.append(
            self.actor_network_optimizer.make_optimizer_scheduler(
                self.actor_network.parameters()
            )
        )
        if self.alpha_optimizer is not None:
            optimizers.append(
                self.alpha_optimizer.make_optimizer_scheduler([self.log_alpha])
            )
        if self.value_network:
            optimizers.append(
                self.value_network_optimizer.make_optimizer_scheduler(
                    self.value_network.parameters()
                )
            )
        # soft-update
        if self.value_network:
            target_params = self.value_network_target.parameters()
            source_params = self.value_network.parameters()
        else:
            target_params = list(self.q1_network_target.parameters())
            source_params = list(self.q1_network.parameters())
            if self.q2_network:
                target_params += list(self.q2_network_target.parameters())
                source_params += list(self.q2_network.parameters())
        optimizers.append(
            SoftUpdate.make_optimizer_scheduler(
                target_params, source_params, tau=self.tau
            )
        )

        return optimizers

    def train_step_gen(self, training_batch: rlt.PolicyNetworkInput, batch_idx: int):
        """
        IMPORTANT: the input action here is assumed to match the
        range of the output of the actor.
        """

        assert isinstance(training_batch, rlt.PolicyNetworkInput)

        state = training_batch.state
        action = training_batch.action
        reward = training_batch.reward
        discount = torch.full_like(reward, self.gamma)
        not_done_mask = training_batch.not_terminal

        #
        # First, optimize Q networks; minimizing MSE between
        # Q(s, a) & r + discount * V'(next_s)
        #

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

        q1_value = self.q1_network(state, action)
        q1_loss = F.mse_loss(q1_value, target_q_value)
        yield q1_loss

        if self.q2_network:
            q2_value = self.q2_network(state, action)
            q2_loss = F.mse_loss(q2_value, target_q_value)
            yield q2_loss

        # Second, optimize the actor; minimizing KL-divergence between
        # propensity & softmax of value.  Due to reparameterization trick,
        # it ends up being log_prob(actor_action) - Q(s, actor_action)

        actor_output = self.actor_network(state)

        state_actor_action = (state, rlt.FeatureData(actor_output.action))
        q1_actor_value = self.q1_network(*state_actor_action)
        min_q_actor_value = q1_actor_value
        if self.q2_network:
            q2_actor_value = self.q2_network(*state_actor_action)
            min_q_actor_value = torch.min(q1_actor_value, q2_actor_value)

        actor_log_prob = actor_output.log_prob

        if not self.backprop_through_log_prob:
            actor_log_prob = actor_log_prob.detach()

        if self.crr_config is not None:
            cur_value = self.value_network(training_batch.state.float_features)
            advantage = (min_q_actor_value - cur_value).detach()
            # pyre-fixme[16]: `Optional` has no attribute `get_weight_from_advantage`.
            crr_weight = self.crr_config.get_weight_from_advantage(advantage)
            assert (
                actor_log_prob.shape == crr_weight.shape
            ), f"{actor_log_prob.shape} != {crr_weight.shape}"
            actor_loss = -(actor_log_prob * crr_weight.detach())
        else:
            actor_loss = self.entropy_temperature * actor_log_prob - min_q_actor_value
        # Do this in 2 steps so we can log histogram of actor loss
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

        yield actor_loss_mean

        # Optimize Alpha
        if self.alpha_optimizer is not None:
            alpha_loss = -(
                (
                    self.log_alpha
                    * (actor_output.log_prob + self.target_entropy).detach()
                ).mean()
            )
            yield alpha_loss
            self.entropy_temperature = self.log_alpha.exp()

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
                log_prob_a = actor_output.log_prob
                log_prob_a = log_prob_a.clamp(-20.0, 20.0)
                target_value = min_q_actor_value - self.entropy_temperature * log_prob_a

            value_loss = F.mse_loss(state_value, target_value.detach())
            yield value_loss

        self.logger.log_metrics(
            {
                "td_loss": q1_loss,
                "logged_rewards": reward.mean(),
                "model_values_on_logged_actions": q1_value.mean(),
                "q1_value": q1_value.mean(),
                "entropy_temperature": self.entropy_temperature,
                "log_prob_a": log_prob_a.mean(),
                "next_state_value": next_state_value.mean(),
                "target_q_value": target_q_value.mean(),
                "min_q_actor_value": min_q_actor_value.mean(),
                "actor_output_log_prob": actor_output.log_prob.mean(),
                "actor_loss": actor_loss.mean(),
            },
            step=self.all_batches_processed,
        )
        if self.q2_network:
            self.logger.log_metrics(
                {"q2_value": q2_value.mean()},
                step=self.all_batches_processed,
            )

        if self.value_network:
            self.logger.log_metrics(
                {"target_state_value": target_value.mean()},
                step=self.all_batches_processed,
            )

        if self.add_kld_to_loss:
            self.logger.log_metrics(
                {
                    "action_batch_mean": action_batch_m.mean(),
                    "action_batch_var": action_batch_v.mean(),
                    "kld": kld,
                },
                step=self.all_batches_processed,
            )

        # Use the soft update rule to update the target networks
        result = self.soft_update_result()
        self.log("td_loss", q1_loss, prog_bar=True)
        yield result
