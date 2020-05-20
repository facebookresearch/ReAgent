#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Tuple

import reagent.parameters as rlp
import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.core.configuration import resolve_defaults
from reagent.core.dataclasses import dataclass, field
from reagent.training.dqn_trainer_base import DQNTrainerBase


logger = logging.getLogger(__name__)


# TODO(T67238979): Make implicit
@dataclass(frozen=True)
class ParametricDQNTrainerParameters:
    __hash__ = rlp.param_hash

    rl: rlp.RLParameters = field(default_factory=rlp.RLParameters)
    double_q_learning: bool = True
    minibatch_size: int = 1024
    minibatches_per_step: int = 1
    optimizer: rlp.OptimizerParameters = field(default_factory=rlp.OptimizerParameters)


class ParametricDQNTrainer(DQNTrainerBase):
    @resolve_defaults
    def __init__(
        self,
        q_network,
        q_network_target,
        reward_network,
        params: ParametricDQNTrainerParameters,
        use_gpu: bool = False,
    ) -> None:
        super().__init__(params.rl, use_gpu=use_gpu)

        self.double_q_learning = params.double_q_learning
        self.minibatch_size = params.minibatch_size
        self.minibatches_per_step = params.minibatches_per_step or 1

        self.q_network = q_network
        self.q_network_target = q_network_target
        self._set_optimizer(params.optimizer.optimizer)
        # pyre-fixme[16]: `ParametricDQNTrainer` has no attribute `optimizer_func`.
        self.q_network_optimizer = self.optimizer_func(
            self.q_network.parameters(),
            lr=params.optimizer.learning_rate,
            weight_decay=params.optimizer.l2_decay,
        )

        self.reward_network = reward_network
        self.reward_network_optimizer = self.optimizer_func(
            self.reward_network.parameters(),
            lr=params.optimizer.learning_rate,
            weight_decay=params.optimizer.l2_decay,
        )

    def warm_start_components(self):
        return [
            "q_network",
            "q_network_target",
            "q_network_optimizer",
            "reward_network",
            "reward_network_optimizer",
        ]

    @torch.no_grad()
    def get_detached_q_values(self, state, action) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Gets the q values from the model and target networks """
        q_values = self.q_network(state, action)
        q_values_target = self.q_network_target(state, action)
        return q_values, q_values_target

    @torch.no_grad()
    def train(self, training_batch: rlt.ParametricDqnInput) -> None:
        self.minibatch += 1
        reward = training_batch.reward
        not_terminal = training_batch.not_terminal.float()
        discount_tensor = torch.full_like(reward, self.gamma)
        if self.use_seq_num_diff_as_time_diff:
            assert self.multi_steps is None
            discount_tensor = torch.pow(self.gamma, training_batch.time_diff.float())
        if self.multi_steps is not None:
            # pyre-fixme[16]: Optional type has no attribute `float`.
            discount_tensor = torch.pow(self.gamma, training_batch.step.float())

        if self.maxq_learning:
            # Assuming actions are parametrized in a k-dimensional space
            # tiled_state = (batch_size * max_num_action, state_dim)
            # possible_actions = (batch_size* max_num_action, k)
            # possible_actions_mask = (batch_size, max_num_action)
            product = training_batch.possible_next_actions.float_features.shape[0]
            batch_size = training_batch.possible_actions_mask.shape[0]
            assert product % batch_size == 0, (
                f"batch_size * max_num_action {product} is "
                f"not divisible by batch_size {batch_size}"
            )
            max_num_action = product // batch_size
            tiled_next_state = training_batch.next_state.get_tiled_batch(max_num_action)
            all_next_q_values, all_next_q_values_target = self.get_detached_q_values(
                tiled_next_state, training_batch.possible_next_actions
            )
            # Compute max a' Q(s', a') over all possible actions using target network
            next_q_values, _ = self.get_max_q_values_with_target(
                all_next_q_values,
                all_next_q_values_target,
                training_batch.possible_next_actions_mask.float(),
            )
            assert (
                len(next_q_values.shape) == 2 and next_q_values.shape[1] == 1
            ), f"{next_q_values.shape}"

        else:
            # SARSA (Use the target network)
            _, next_q_values = self.get_detached_q_values(
                training_batch.next_state, training_batch.next_action
            )
            assert (
                len(next_q_values.shape) == 2 and next_q_values.shape[1] == 1
            ), f"{next_q_values.shape}"

        target_q_values = reward + not_terminal * discount_tensor * next_q_values
        assert (
            target_q_values.shape[-1] == 1
        ), f"{target_q_values.shape} doesn't end with 1"

        with torch.enable_grad():
            # Get Q-value of action taken
            q_values = self.q_network(training_batch.state, training_batch.action)
            assert (
                target_q_values.shape == q_values.shape
            ), f"{target_q_values.shape} != {q_values.shape}."
            td_loss = self.q_network_loss(q_values, target_q_values)
            td_loss.backward()
            self._maybe_run_optimizer(
                self.q_network_optimizer, self.minibatches_per_step
            )

        # Use the soft update rule to update target network
        self._maybe_soft_update(
            self.q_network, self.q_network_target, self.tau, self.minibatches_per_step
        )

        with torch.enable_grad():
            # pyre-fixme[16]: Optional type has no attribute `metrics`.
            if training_batch.extras.metrics is not None:
                metrics_reward_concat_real_vals = torch.cat(
                    (reward, training_batch.extras.metrics), dim=1
                )
            else:
                metrics_reward_concat_real_vals = reward

            # get reward estimates
            reward_estimates = self.reward_network(
                training_batch.state, training_batch.action
            )
            reward_loss = F.mse_loss(
                reward_estimates, metrics_reward_concat_real_vals.squeeze(-1)
            )
            reward_loss.backward()
            self._maybe_run_optimizer(
                self.reward_network_optimizer, self.minibatches_per_step
            )

        self.loss_reporter.report(
            td_loss=td_loss.detach().cpu(),
            reward_loss=reward_loss.detach().cpu(),
            logged_rewards=reward,
            model_values_on_logged_actions=q_values.detach().cpu(),
        )
