#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List, Tuple

import reagent.core.types as rlt
import torch
from reagent.core.configuration import resolve_defaults
from reagent.core.dataclasses import field
from reagent.core.parameters import EvaluationParameters, RLParameters
from reagent.optimizer import SoftUpdate
from reagent.optimizer.union import Optimizer__Union
from reagent.training.dqn_trainer_base import DQNTrainerBaseLightning


logger = logging.getLogger(__name__)


class QRDQNTrainer(DQNTrainerBaseLightning):
    """
    Implementation of QR-DQN (Quantile Regression Deep Q-Network)

    See https://arxiv.org/abs/1710.10044 for details
    """

    @resolve_defaults
    def __init__(
        self,
        q_network,
        q_network_target,
        metrics_to_score=None,
        reward_network=None,
        q_network_cpe=None,
        q_network_cpe_target=None,
        actions: List[str] = field(default_factory=list),  # noqa: B008
        rl: RLParameters = field(default_factory=RLParameters),  # noqa: B008
        double_q_learning: bool = True,
        num_atoms: int = 51,
        minibatch_size: int = 1024,
        minibatches_per_step: int = 1,
        optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        cpe_optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        evaluation: EvaluationParameters = field(  # noqa: B008
            default_factory=EvaluationParameters
        ),
    ) -> None:
        super().__init__(
            rl_parameters=rl,
            metrics_to_score=metrics_to_score,
            actions=actions,
            evaluation_parameters=evaluation,
        )
        # TODO: check to ensure no rl parameter value is set that isn't actively used by class
        self.double_q_learning = double_q_learning
        self.minibatch_size = minibatch_size
        self.minibatches_per_step = minibatches_per_step
        self._actions = actions

        self.q_network = q_network
        self.q_network_target = q_network_target
        self.q_network_optimizer = optimizer

        self.num_atoms = num_atoms
        self.register_buffer("quantiles", None)
        self.quantiles = (
            (0.5 + torch.arange(self.num_atoms).float()) / float(self.num_atoms)
        ).view(1, -1)

        self._initialize_cpe(
            reward_network, q_network_cpe, q_network_cpe_target, optimizer=cpe_optimizer
        )

    def configure_optimizers(self):
        optimizers = []
        target_params = list(self.q_network_target.parameters())
        source_params = list(self.q_network.parameters())

        optimizers.append(
            self.q_network_optimizer.make_optimizer_scheduler(
                self.q_network.parameters()
            )
        )

        if self.calc_cpe_in_training:
            (
                cpe_target_params,
                cpe_source_params,
                cpe_optimizers,
            ) = self._configure_cpe_optimizers()
            target_params += cpe_target_params
            source_params += cpe_source_params
            optimizers += cpe_optimizers

        optimizers.append(
            SoftUpdate.make_optimizer_scheduler(
                target_params, source_params, tau=self.tau
            )
        )

        return optimizers

    def train_step_gen(self, training_batch: rlt.DiscreteDqnInput, batch_idx: int):
        self._check_input(training_batch)

        rewards = self.boost_rewards(training_batch.reward, training_batch.action)
        discount_tensor = torch.full_like(rewards, self.gamma)
        possible_next_actions_mask = training_batch.possible_next_actions_mask.float()
        possible_actions_mask = training_batch.possible_actions_mask.float()

        not_done_mask = training_batch.not_terminal.float()

        if self.use_seq_num_diff_as_time_diff:
            assert self.multi_steps is None
            discount_tensor = torch.pow(self.gamma, training_batch.time_diff.float())
        if self.multi_steps is not None:
            assert training_batch.step is not None
            discount_tensor = torch.pow(self.gamma, training_batch.step.float())

        next_qf = self.q_network_target(training_batch.next_state)

        if self.maxq_learning:
            # Select distribution corresponding to max valued action
            next_q_values = (
                self.q_network(training_batch.next_state)
                if self.double_q_learning
                else next_qf
            ).mean(dim=2)
            next_action = self.argmax_with_mask(
                next_q_values, possible_next_actions_mask
            )
            next_qf = next_qf[range(rewards.shape[0]), next_action.reshape(-1)]
        else:
            next_qf = (next_qf * training_batch.next_action.unsqueeze(-1)).sum(1)

        # Build target distribution
        target_Q = rewards + discount_tensor * not_done_mask * next_qf

        current_qf = self.q_network(training_batch.state)

        # for reporting only
        all_q_values = current_qf.mean(2).detach()

        current_qf = (current_qf * training_batch.action.unsqueeze(-1)).sum(1)

        # (batch, atoms) -> (atoms, batch, 1) -> (atoms, batch, atoms)
        td = target_Q.t().unsqueeze(-1) - current_qf
        loss = (
            self.huber(td) * (self.quantiles - (td.detach() < 0).float()).abs()
        ).mean()

        yield loss
        # pyre-fixme[16]: `DQNTrainer` has no attribute `loss`.
        self.loss = loss.detach()

        # Get Q-values of next states, used in computing cpe
        all_next_action_scores = (
            self.q_network(training_batch.next_state).detach().mean(dim=2)
        )

        logged_action_idxs = torch.argmax(training_batch.action, dim=1, keepdim=True)
        yield from self._calculate_cpes(
            training_batch,
            training_batch.state,
            training_batch.next_state,
            all_q_values,
            all_next_action_scores,
            logged_action_idxs,
            discount_tensor,
            not_done_mask,
        )

        model_action_idxs = self.argmax_with_mask(
            all_q_values,
            possible_actions_mask if self.maxq_learning else training_batch.action,
        )

        self.reporter.log(
            td_loss=loss,
            logged_actions=logged_action_idxs,
            logged_propensities=training_batch.extras.action_probability,
            logged_rewards=rewards,
            logged_values=None,  # Compute at end of each epoch for CPE
            model_values=all_q_values,
            model_values_on_logged_actions=None,  # Compute at end of each epoch for CPE
            model_action_idxs=model_action_idxs,
        )

        yield self.soft_update_result()

    @torch.no_grad()
    def boost_rewards(
        self, rewards: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        # Apply reward boost if specified
        reward_boosts = torch.sum(
            actions.float() * self.reward_boosts, dim=1, keepdim=True
        )
        return rewards + reward_boosts

    def argmax_with_mask(self, q_values, possible_actions_mask):
        # Set q-values of impossible actions to a very large negative number.
        q_values = q_values.reshape(possible_actions_mask.shape)
        q_values = q_values + self.ACTION_NOT_POSSIBLE_VAL * (1 - possible_actions_mask)
        return q_values.argmax(1)

    # Used to prevent warning when a.shape != b.shape
    def huber(self, x):
        return torch.where(x.abs() < 1, 0.5 * x.pow(2), x.abs() - 0.5)

    @torch.no_grad()
    def get_detached_model_outputs(
        self, state: rlt.FeatureData
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets the q values from the model and target networks"""
        q_values = self.q_network(state).mean(dim=2)
        q_values_target = self.q_network_target(state).mean(dim=2)
        return q_values, q_values_target
