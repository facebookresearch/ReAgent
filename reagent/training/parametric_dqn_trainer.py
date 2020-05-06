#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Tuple

import reagent.parameters as rlp
import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.core.configuration import make_config_class, resolve_defaults
from reagent.core.dataclasses import field
from reagent.training.dqn_trainer_base import DQNTrainerBase
from reagent.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


class ParametricDQNTrainer(DQNTrainerBase):
    @resolve_defaults
    def __init__(
        self,
        q_network,
        q_network_target,
        reward_network,
        rl: rlp.RLParameters = field(default_factory=rlp.RLParameters),  # noqa B008
        double_q_learning: bool = True,
        minibatch_size: int = 1024,
        minibatches_per_step: int = 1,
        optimizer: rlp.OptimizerParameters = field(  # noqa B008
            default_factory=rlp.OptimizerParameters
        ),
        use_gpu: bool = False,
    ) -> None:
        super().__init__(rl, use_gpu=use_gpu)

        self.double_q_learning = double_q_learning
        self.minibatch_size = minibatch_size
        self.minibatches_per_step = minibatches_per_step or 1

        self.q_network = q_network
        self.q_network_target = q_network_target
        self._set_optimizer(optimizer.optimizer)
        # pyre-fixme[16]: `ParametricDQNTrainer` has no attribute `optimizer_func`.
        self.q_network_optimizer = self.optimizer_func(
            self.q_network.parameters(),
            lr=optimizer.learning_rate,
            weight_decay=optimizer.l2_decay,
        )

        self.reward_network = reward_network
        self.reward_network_optimizer = self.optimizer_func(
            self.reward_network.parameters(),
            lr=optimizer.learning_rate,
            weight_decay=optimizer.l2_decay,
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
    def get_detached_q_values(
        self, state, action
    ) -> Tuple[rlt.SingleQValue, rlt.SingleQValue]:
        """ Gets the q values from the model and target networks """
        input = rlt.PreprocessedStateAction(state=state, action=action)
        q_values = self.q_network(input)
        q_values_target = self.q_network_target(input)
        return q_values.q_value, q_values_target.q_value

    @torch.no_grad()
    def train(self, training_batch) -> None:
        if isinstance(training_batch, TrainingDataPage):
            training_batch = training_batch.as_parametric_maxq_training_batch()

        learning_input = training_batch.training_input
        self.minibatch += 1

        reward = learning_input.reward
        not_done_mask = learning_input.not_terminal

        discount_tensor = torch.full_like(reward, self.gamma)
        if self.use_seq_num_diff_as_time_diff:
            assert self.multi_steps is None
            discount_tensor = torch.pow(self.gamma, learning_input.time_diff.float())
        if self.multi_steps is not None:
            discount_tensor = torch.pow(self.gamma, learning_input.step.float())

        if self.maxq_learning:
            all_next_q_values, all_next_q_values_target = self.get_detached_q_values(
                learning_input.tiled_next_state, learning_input.possible_next_actions
            )
            # Compute max a' Q(s', a') over all possible actions using target network
            next_q_values, _ = self.get_max_q_values_with_target(
                all_next_q_values,
                all_next_q_values_target,
                learning_input.possible_next_actions_mask.float(),
            )
        else:
            # SARSA (Use the target network)
            _, next_q_values = self.get_detached_q_values(
                learning_input.next_state, learning_input.next_action
            )

        filtered_max_q_vals = next_q_values * not_done_mask.float()

        target_q_values = reward + (discount_tensor * filtered_max_q_vals)

        with torch.enable_grad():
            # Get Q-value of action taken
            current_state_action = rlt.PreprocessedStateAction(
                state=learning_input.state, action=learning_input.action
            )
            q_values = self.q_network(current_state_action).q_value
            # pyre-fixme[16]: `ParametricDQNTrainer` has no attribute
            #  `all_action_scores`.
            self.all_action_scores = q_values.detach()

            value_loss = self.q_network_loss(q_values, target_q_values)
            # pyre-fixme[16]: `ParametricDQNTrainer` has no attribute `loss`.
            self.loss = value_loss.detach()
            value_loss.backward()
            self._maybe_run_optimizer(
                self.q_network_optimizer, self.minibatches_per_step
            )

        # Use the soft update rule to update target network
        self._maybe_soft_update(
            self.q_network, self.q_network_target, self.tau, self.minibatches_per_step
        )

        with torch.enable_grad():
            if training_batch.extras.metrics is not None:
                metrics_reward_concat_real_vals = torch.cat(
                    (reward, training_batch.extras.metrics), dim=1
                )
            else:
                metrics_reward_concat_real_vals = reward
            # get reward estimates
            reward_estimates = self.reward_network(current_state_action).q_value
            reward_loss = F.mse_loss(reward_estimates, metrics_reward_concat_real_vals)
            reward_loss.backward()
            self._maybe_run_optimizer(
                self.reward_network_optimizer, self.minibatches_per_step
            )

        self.loss_reporter.report(
            td_loss=self.loss,
            reward_loss=reward_loss,
            logged_rewards=reward,
            model_values_on_logged_actions=self.all_action_scores,
        )

    @torch.no_grad()
    def internal_prediction(self, state, action):
        """
        Only used by Gym
        """
        self.q_network.eval()
        q_values = self.q_network(
            rlt.PreprocessedStateAction.from_tensors(state=state, action=action)
        )
        self.q_network.train()
        return q_values.q_value.cpu()

    @torch.no_grad()
    def internal_reward_estimation(self, state, action):
        """
        Only used by Gym
        """
        self.reward_network.eval()
        reward_estimates = self.reward_network(
            rlt.PreprocessedStateAction.from_tensors(state=state, action=action)
        )
        self.reward_network.train()
        return reward_estimates.q_value.cpu()


@make_config_class(ParametricDQNTrainer.__init__, blacklist=["use_gpu"])
class ParametricDQNTrainerParameters:
    pass
