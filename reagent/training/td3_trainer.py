#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import copy
import logging
from typing import Optional

import reagent.types as rlt
import torch
from reagent.core.dataclasses import dataclass, field
from reagent.parameters import (
    CONTINUOUS_TRAINING_ACTION_RANGE,
    OptimizerParameters,
    RLParameters,
    param_hash,
)
from reagent.tensorboardX import SummaryWriterContext
from reagent.training.rl_trainer_pytorch import RLTrainer
from reagent.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TD3TrainingParameters:
    __hash__ = param_hash

    rl: RLParameters = field(default_factory=RLParameters)
    minibatch_size: int = 64
    q_network_optimizer: OptimizerParameters = OptimizerParameters()
    actor_network_optimizer: OptimizerParameters = OptimizerParameters()
    use_2_q_functions: bool = True
    noise_variance: float = 0.2
    noise_clip: float = 0.5
    delayed_policy_update: int = 2
    warm_start_model_path: Optional[str] = None
    minibatches_per_step: int = 1


class TD3Trainer(RLTrainer):
    """
    Twin Delayed Deep Deterministic Policy Gradient algorithm trainer
    as described in https://arxiv.org/pdf/1802.09477
    """

    def __init__(
        self,
        q1_network,
        actor_network,
        parameters: TD3TrainingParameters,
        use_gpu: bool = False,
        q2_network=None,
    ) -> None:
        """
        Args:
            The four args below are provided for integration with other
            environments (e.g., Gym):
        """
        super().__init__(parameters.rl, use_gpu=use_gpu)

        self.minibatch_size = parameters.minibatch_size
        self.minibatches_per_step = parameters.minibatches_per_step or 1

        self.q1_network = q1_network
        self.q1_network_target = copy.deepcopy(self.q1_network)
        self.q1_network_optimizer = self._get_optimizer(
            q1_network, parameters.q_network_optimizer
        )

        self.q2_network = q2_network
        if self.q2_network is not None:
            self.q2_network_target = copy.deepcopy(self.q2_network)
            self.q2_network_optimizer = self._get_optimizer(
                q2_network, parameters.q_network_optimizer
            )

        self.actor_network = actor_network
        self.actor_network_target = copy.deepcopy(self.actor_network)
        self.actor_network_optimizer = self._get_optimizer(
            actor_network, parameters.actor_network_optimizer
        )

        self.noise_variance = parameters.noise_variance
        self.noise_clip_range = (-parameters.noise_clip, parameters.noise_clip)
        self.delayed_policy_update = parameters.delayed_policy_update

    def warm_start_components(self):
        components = [
            "q1_network",
            "q1_network_target",
            "q1_network_optimizer",
            "actor_network",
            "actor_network_target",
            "actor_network_optimizer",
        ]
        if self.q2_network:
            components += ["q2_network", "q2_network_target", "q2_network_optimizer"]

        return components

    def train(self, training_batch: rlt.PolicyNetworkInput) -> None:
        """
        IMPORTANT: the input action here is assumed to be preprocessed to match the
        range of the output of the actor.
        """
        if isinstance(training_batch, TrainingDataPage):
            training_batch = training_batch.as_policy_network_training_batch()

        assert isinstance(training_batch, rlt.PolicyNetworkInput)

        self.minibatch += 1

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
        # NOTE: important to zero here (instead of using _maybe_update)
        # since q1 may have accumulated gradients from actor network update
        self.q1_network_optimizer.zero_grad()
        q1_value = self.q1_network(state, action)
        q1_loss = self.q_network_loss(q1_value, target_q_value)
        q1_loss.backward()
        self.q1_network_optimizer.step()

        if self.q2_network:
            self.q2_network_optimizer.zero_grad()
            q2_value = self.q2_network(state, action)
            q2_loss = self.q_network_loss(q2_value, target_q_value)
            q2_loss.backward()
            self.q2_network_optimizer.step()

        # Only update actor and target networks after a fixed number of Q updates
        if self.minibatch % self.delayed_policy_update == 0:
            self.actor_network_optimizer.zero_grad()
            actor_action = self.actor_network(state).action
            actor_q1_value = self.q1_network(state, rlt.FeatureData(actor_action))
            actor_loss = -(actor_q1_value.mean())
            actor_loss.backward()
            self.actor_network_optimizer.step()

            self._soft_update(self.q1_network, self.q1_network_target, self.tau)
            self._soft_update(self.q2_network, self.q2_network_target, self.tau)
            self._soft_update(self.actor_network, self.actor_network_target, self.tau)

        # Logging at the end to schedule all the cuda operations first
        if (
            self.tensorboard_logging_freq != 0
            and self.minibatch % self.tensorboard_logging_freq == 0
        ):
            logs = {
                "loss/q1_loss": q1_loss,
                "loss/actor_loss": actor_loss,
                "q_value/q1_value": q1_value,
                "q_value/next_q_value": next_q_value,
                "q_value/target_q_value": target_q_value,
                "q_value/actor_q1_value": actor_q1_value,
            }
            if self.q2_network:
                logs.update({"loss/q2_loss": q2_loss, "q_value/q2_value": q2_value})

            for k, v in logs.items():
                v = v.detach().cpu()
                if v.dim() == 0:
                    # pyre-fixme[16]: `SummaryWriterContext` has no attribute
                    #  `add_scalar`.
                    SummaryWriterContext.add_scalar(k, v.item())
                    continue

                elif v.dim() == 2:
                    v = v.squeeze(1)
                assert v.dim() == 1
                SummaryWriterContext.add_histogram(k, v.numpy())
                SummaryWriterContext.add_scalar(f"{k}_mean", v.mean().item())

        self.loss_reporter.report(
            td_loss=float(q1_loss),
            reward_loss=None,
            logged_rewards=reward,
            model_values_on_logged_actions=q1_value,
        )
