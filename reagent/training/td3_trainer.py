#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import copy
import logging

import reagent.core.types as rlt
import torch
import torch.nn.functional as F
from reagent.core.configuration import resolve_defaults
from reagent.core.dataclasses import field
from reagent.core.parameters import CONTINUOUS_TRAINING_ACTION_RANGE, RLParameters
from reagent.optimizer import Optimizer__Union, SoftUpdate
from reagent.training.reagent_lightning_module import ReAgentLightningModule
from reagent.training.rl_trainer_pytorch import RLTrainerMixin


logger = logging.getLogger(__name__)


class TD3Trainer(RLTrainerMixin, ReAgentLightningModule):
    """
    Twin Delayed Deep Deterministic Policy Gradient algorithm trainer
    as described in https://arxiv.org/pdf/1802.09477
    """

    @resolve_defaults
    def __init__(
        self,
        actor_network,
        q1_network,
        q2_network=None,
        # Start TD3TrainerParameters
        rl: RLParameters = field(default_factory=RLParameters),  # noqa: B008
        q_network_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        actor_network_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        noise_variance: float = 0.2,
        noise_clip: float = 0.5,
        delayed_policy_update: int = 2,
    ) -> None:
        """
        Args:
            actor_network: states -> actions, trained to maximize value
            q1_network: states, action -> q-value
            q2_network (optional): double q-learning to stabilize training
                from overestimation bias
            rl (optional): an instance of the RLParameter class, which
                defines relevant hyperparameters
            q_network_optimizer (optional): the optimizer class and
                optimizer hyperparameters for the q network(s) optimizer
            actor_network_optimizer (optional): see q_network_optimizer
            noise_variance (optional): the variance of action noise added to smooth
                q-value estimates
            noise_clip (optional): the maximum absolute value of action noise added
                to smooth q-value estimates
            delayed_policy_update (optional): the ratio of q network updates
                to target and policy network updates
        """
        super().__init__()
        self.rl_parameters = rl

        self.q1_network = q1_network
        self.q1_network_target = copy.deepcopy(self.q1_network)
        self.q_network_optimizer = q_network_optimizer

        self.q2_network = q2_network
        if self.q2_network is not None:
            self.q2_network_target = copy.deepcopy(self.q2_network)

        self.actor_network = actor_network
        self.actor_network_target = copy.deepcopy(self.actor_network)
        self.actor_network_optimizer = actor_network_optimizer

        self.noise_variance = noise_variance
        self.noise_clip_range = (-noise_clip, noise_clip)
        self.delayed_policy_update = delayed_policy_update

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

        # soft-update
        target_params = list(self.q1_network_target.parameters())
        source_params = list(self.q1_network.parameters())
        if self.q2_network:
            target_params += list(self.q2_network_target.parameters())
            source_params += list(self.q2_network.parameters())
        target_params += list(self.actor_network_target.parameters())
        source_params += list(self.actor_network.parameters())
        optimizers.append(
            SoftUpdate.make_optimizer_scheduler(
                target_params, source_params, tau=self.tau
            )
        )

        return optimizers

    def train_step_gen(self, training_batch: rlt.PolicyNetworkInput, batch_idx: int):
        """
        IMPORTANT: the input action here is assumed to be preprocessed to match the
        range of the output of the actor.
        """
        assert isinstance(training_batch, rlt.PolicyNetworkInput)

        state = training_batch.state
        action = training_batch.action
        next_state = training_batch.next_state
        reward = training_batch.reward
        not_terminal = training_batch.not_terminal

        # Generate target = r + y * min (Q1(s',pi(s')), Q2(s',pi(s')))
        with torch.no_grad():
            next_actor = self.actor_network_target(next_state).action
            noise = torch.randn_like(next_actor) * self.noise_variance
            next_actor = (next_actor + noise.clamp(*self.noise_clip_range)).clamp(
                *CONTINUOUS_TRAINING_ACTION_RANGE
            )
            next_state_actor = (next_state, rlt.FeatureData(next_actor))
            next_q_value = self.q1_network_target(*next_state_actor)

            if self.q2_network is not None:
                next_q_value = torch.min(
                    next_q_value, self.q2_network_target(*next_state_actor)
                )

            target_q_value = reward + self.gamma * next_q_value * not_terminal.float()

        # Optimize Q1 and Q2
        q1_value = self.q1_network(state, action)
        q1_loss = F.mse_loss(q1_value, target_q_value)
        if batch_idx % self.trainer.log_every_n_steps == 0:
            self.reporter.log(
                q1_loss=q1_loss,
                q1_value=q1_value,
                next_q_value=next_q_value,
                target_q_value=target_q_value,
            )
        self.log(
            "td_loss", q1_loss, prog_bar=True, batch_size=training_batch.batch_size()
        )
        yield q1_loss

        if self.q2_network:
            q2_value = self.q2_network(state, action)
            q2_loss = F.mse_loss(q2_value, target_q_value)
            if batch_idx % self.trainer.log_every_n_steps == 0:
                self.reporter.log(
                    q2_loss=q2_loss,
                    q2_value=q2_value,
                )
            yield q2_loss

        # Only update actor and target networks after a fixed number of Q updates
        if batch_idx % self.delayed_policy_update == 0:
            actor_action = self.actor_network(state).action
            actor_q1_value = self.q1_network(state, rlt.FeatureData(actor_action))
            actor_loss = -(actor_q1_value.mean())
            if batch_idx % self.trainer.log_every_n_steps == 0:
                self.reporter.log(
                    actor_loss=actor_loss,
                    actor_q1_value=actor_q1_value,
                )
            yield actor_loss

            # Use the soft update rule to update the target networks
            result = self.soft_update_result()
            yield result

        else:
            # Yielding None prevents the actor and target networks from updating
            yield None
            yield None
