#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging
from typing import Optional, Tuple

import reagent.core.parameters as rlp
import reagent.core.types as rlt
import torch
import torch.nn as nn
import torch.nn.functional as F
from reagent.core.configuration import resolve_defaults
from reagent.core.dataclasses import field
from reagent.optimizer import Optimizer__Union, SoftUpdate
from reagent.training.dqn_trainer_base import DQNTrainerMixin
from reagent.training.reagent_lightning_module import ReAgentLightningModule
from reagent.training.rl_trainer_pytorch import RLTrainerMixin

logger = logging.getLogger(__name__)


class ParametricDQNTrainer(DQNTrainerMixin, RLTrainerMixin, ReAgentLightningModule):
    @resolve_defaults
    def __init__(
        self,
        q_network,
        q_network_target,
        reward_network: Optional[nn.Module] = None,
        # Start ParametricDQNTrainerParameters
        rl: rlp.RLParameters = field(default_factory=rlp.RLParameters),  # noqa: B008
        double_q_learning: bool = True,
        minibatches_per_step: int = 1,
        optimizer: Optimizer__Union = field(  # noqa: B008
            default_factory=Optimizer__Union.default
        ),
        log_tensorboard: bool = False,
    ) -> None:
        super().__init__()
        self.rl_parameters = rl

        self.double_q_learning = double_q_learning
        self.minibatches_per_step = minibatches_per_step or 1

        self.q_network = q_network
        self.q_network_target = q_network_target
        self.reward_network = reward_network
        self.optimizer = optimizer
        self.log_tensorboard = log_tensorboard

        if rl.q_network_loss == "mse":
            self.q_network_loss = F.mse_loss
        elif rl.q_network_loss == "huber":
            self.q_network_loss = F.smooth_l1_loss
        elif rl.q_network_loss == "bce_with_logits":
            # The loss is only used when gamma = 0, reward is between 0 and 1
            # and we need to calculate NE as metrics.
            assert rl.gamma == 0, (
                "bce_with_logits loss is only supported when gamma is 0."
            )
            self.q_network_loss = F.binary_cross_entropy_with_logits
        else:
            raise Exception(
                "Q-Network loss type {} not valid loss.".format(rl.q_network_loss)
            )

    def configure_optimizers(self):
        optimizers = []
        optimizers.append(
            self.optimizer.make_optimizer_scheduler(self.q_network.parameters())
        )
        if self.reward_network is not None:
            optimizers.append(
                self.optimizer.make_optimizer_scheduler(
                    self.reward_network.parameters()
                )
            )
        # soft-update
        target_params = list(self.q_network_target.parameters())
        source_params = list(self.q_network.parameters())
        optimizers.append(
            SoftUpdate.make_optimizer_scheduler(
                target_params, source_params, tau=self.tau
            )
        )

        return optimizers

    def _check_input(self, training_batch: rlt.ParametricDqnInput):
        assert isinstance(training_batch, rlt.ParametricDqnInput)
        assert training_batch.not_terminal.dim() == training_batch.reward.dim() == 2
        assert (
            training_batch.not_terminal.shape[1] == training_batch.reward.shape[1] == 1
        )
        assert (
            training_batch.action.float_features.dim()
            == training_batch.next_action.float_features.dim()
            == 2
        )

    @torch.no_grad()
    def get_detached_model_outputs(
        self, state, action
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets the q values from the model and target networks"""
        q_values = self.q_network(state, action)
        q_values_target = self.q_network_target(state, action)
        return q_values, q_values_target

    def train_step_gen(self, training_batch: rlt.ParametricDqnInput, batch_idx: int):
        self._check_input(training_batch)
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
            (
                all_next_q_values,
                all_next_q_values_target,
            ) = self.get_detached_model_outputs(
                tiled_next_state, training_batch.possible_next_actions
            )
            # Compute max a' Q(s', a') over all possible actions using target network
            next_q_values, _ = self.get_max_q_values_with_target(
                all_next_q_values,
                all_next_q_values_target,
                training_batch.possible_next_actions_mask.float(),
            )
            assert len(next_q_values.shape) == 2 and next_q_values.shape[1] == 1, (
                f"{next_q_values.shape}"
            )

        else:
            # SARSA (Use the target network)
            _, next_q_values = self.get_detached_model_outputs(
                training_batch.next_state, training_batch.next_action
            )
            assert len(next_q_values.shape) == 2 and next_q_values.shape[1] == 1, (
                f"{next_q_values.shape}"
            )

        target_q_values = reward + not_terminal * discount_tensor * next_q_values
        assert target_q_values.shape[-1] == 1, (
            f"{target_q_values.shape} doesn't end with 1"
        )

        # Get Q-value of action taken
        q_values = self.q_network(training_batch.state, training_batch.action)
        assert target_q_values.shape == q_values.shape, (
            f"{target_q_values.shape} != {q_values.shape}."
        )
        td_loss = self.q_network_loss(q_values, target_q_values)
        yield td_loss

        # pyre-fixme[16]: Optional type has no attribute `metrics`.
        if training_batch.extras.metrics is not None:
            metrics_reward_concat_real_vals = torch.cat(
                (reward, training_batch.extras.metrics), dim=1
            )
        else:
            metrics_reward_concat_real_vals = reward

        # get reward estimates
        if self.reward_network is not None:
            reward_estimates = self.reward_network(
                training_batch.state, training_batch.action
            )
            reward_loss = F.mse_loss(
                reward_estimates.squeeze(-1),
                metrics_reward_concat_real_vals.squeeze(-1),
            )
            yield reward_loss
        else:
            reward_loss = torch.tensor([0.0])

        td_loss = td_loss.detach().cpu()
        reward_loss = reward_loss.detach().cpu()
        q_values = q_values.detach().cpu()
        # Logging loss, rewards, and model values
        # Use reagent reporter
        self.reporter.log(
            td_loss=td_loss,
            reward_loss=reward_loss,
            logged_rewards=reward,
            model_values_on_logged_actions=q_values,
        )
        # Use pytorch-lightning logger on rank 0
        # pyre-fixme[16]: Optional type has no attribute `experiment`.
        if self.log_tensorboard and self.logger.experiment:
            self.log("loss", {"td_loss": td_loss, "reward_loss": reward_loss})
            tensorboard = self.logger.experiment
            tensorboard.add_histogram("reward", reward)
            tensorboard.add_histogram("model_values_on_logged_actions", q_values)

        # Use the soft update rule to update target network
        yield self.soft_update_result()
