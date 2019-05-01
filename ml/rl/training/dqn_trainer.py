#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Optional, Tuple

import ml.rl.types as rlt
import numpy as np
import torch
import torch.nn.functional as F
from ml.rl.caffe_utils import masked_softmax
from ml.rl.thrift.core.ttypes import DiscreteActionModelParameters
from ml.rl.training.dqn_trainer_base import DQNTrainerBase
from ml.rl.training.imitator_training import get_valid_actions_from_imitator
from ml.rl.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


class DQNTrainer(DQNTrainerBase):
    def __init__(
        self,
        q_network,
        q_network_target,
        reward_network,
        parameters: DiscreteActionModelParameters,
        use_gpu=False,
        q_network_cpe=None,
        q_network_cpe_target=None,
        metrics_to_score=None,
        imitator=None,
    ) -> None:
        self.double_q_learning = parameters.rainbow.double_q_learning
        self.minibatch_size = parameters.training.minibatch_size
        self.minibatches_per_step = parameters.training.minibatches_per_step or 1
        self._actions = parameters.actions if parameters.actions is not None else []

        DQNTrainerBase.__init__(
            self,
            parameters,
            use_gpu=use_gpu,
            metrics_to_score=metrics_to_score,
            actions=parameters.actions,
        )

        self.q_network = q_network
        self.q_network_target = q_network_target
        self._set_optimizer(parameters.training.optimizer)
        self.q_network_optimizer = self.optimizer_func(
            self.q_network.parameters(),
            lr=parameters.training.learning_rate,
            weight_decay=parameters.training.l2_decay,
        )

        if self.calc_cpe_in_training:
            assert reward_network is not None, "reward_network is required for CPE"
            self.reward_network = reward_network
            self.reward_network_optimizer = self.optimizer_func(
                self.reward_network.parameters(), lr=parameters.training.learning_rate
            )
            assert (
                q_network_cpe is not None and q_network_cpe_target is not None
            ), "q_network_cpe and q_network_cpe_target are required for CPE"
            self.q_network_cpe = q_network_cpe
            self.q_network_cpe_target = q_network_cpe_target
            self.q_network_cpe_optimizer = self.optimizer_func(
                self.q_network_cpe.parameters(), lr=parameters.training.learning_rate
            )
            num_output_nodes = len(self.metrics_to_score) * self.num_actions
            self.reward_idx_offsets = torch.arange(
                0, num_output_nodes, self.num_actions
            ).type(self.dtypelong)

        self.reward_boosts = torch.zeros([1, len(self._actions)]).type(self.dtype)
        if parameters.rl.reward_boost is not None:
            for k in parameters.rl.reward_boost.keys():
                i = self._actions.index(k)
                self.reward_boosts[0, i] = parameters.rl.reward_boost[k]

        # Batch constrained q-learning
        self.bcq = parameters.rainbow.bcq
        if self.bcq:
            self.bcq_drop_threshold = parameters.rainbow.bcq_drop_threshold
            self.bcq_imitator = imitator

    @property
    def num_actions(self) -> int:
        return len(self._actions)

    def get_detached_q_values(
        self, state
    ) -> Tuple[rlt.AllActionQValues, Optional[rlt.AllActionQValues]]:
        """ Gets the q values from the model and target networks """
        with torch.no_grad():
            input = rlt.StateInput(state=state)
            q_values = self.q_network(input).q_values
            q_values_target = self.q_network_target(input).q_values
        return q_values, q_values_target

    def train(self, training_batch):
        if isinstance(training_batch, TrainingDataPage):
            if self.maxq_learning:
                training_batch = training_batch.as_discrete_maxq_training_batch()
            else:
                training_batch = training_batch.as_discrete_sarsa_training_batch()

        learning_input = training_batch.training_input
        boosted_rewards = self.boost_rewards(
            learning_input.reward, learning_input.action
        )

        self.minibatch += 1
        rewards = boosted_rewards
        discount_tensor = torch.full_like(rewards, self.gamma)
        not_done_mask = learning_input.not_terminal.float()

        if self.use_seq_num_diff_as_time_diff:
            assert self.multi_steps is None
            discount_tensor = torch.pow(self.gamma, learning_input.time_diff.float())
        if self.multi_steps is not None:
            discount_tensor = torch.pow(self.gamma, learning_input.step.float())

        all_next_q_values, all_next_q_values_target = self.get_detached_q_values(
            learning_input.next_state
        )

        if self.maxq_learning:
            # Compute max a' Q(s', a') over all possible actions using target network
            possible_next_actions_mask = (
                learning_input.possible_next_actions_mask.float()
            )
            if self.bcq:
                action_on_policy = get_valid_actions_from_imitator(
                    self.bcq_imitator,
                    learning_input.next_state.float_features,
                    self.bcq_drop_threshold,
                )
                possible_next_actions_mask *= action_on_policy
            next_q_values, max_q_action_idxs = self.get_max_q_values_with_target(
                all_next_q_values, all_next_q_values_target, possible_next_actions_mask
            )
        else:
            # SARSA
            next_q_values, max_q_action_idxs = self.get_max_q_values_with_target(
                all_next_q_values, all_next_q_values_target, learning_input.next_action
            )

        filtered_next_q_vals = next_q_values * not_done_mask

        target_q_values = rewards + (discount_tensor * filtered_next_q_vals)

        # Get Q-value of action taken
        current_state = rlt.StateInput(state=learning_input.state)
        all_q_values = self.q_network(current_state).q_values
        self.all_action_scores = all_q_values.detach()
        q_values = torch.sum(all_q_values * learning_input.action, 1, keepdim=True)

        loss = self.q_network_loss(q_values, target_q_values)
        self.loss = loss.detach()

        loss.backward()
        self._maybe_run_optimizer(self.q_network_optimizer, self.minibatches_per_step)

        # Use the soft update rule to update target network
        self._maybe_soft_update(
            self.q_network, self.q_network_target, self.tau, self.minibatches_per_step
        )

        # Get Q-values of next states, used in computing cpe
        with torch.no_grad():
            next_state = rlt.StateInput(state=learning_input.next_state)
            all_next_action_scores = self.q_network(next_state).q_values.detach()

        logged_action_idxs = learning_input.action.argmax(dim=1, keepdim=True)
        reward_loss, model_rewards, model_propensities = self.calculate_cpes(
            training_batch,
            current_state,
            next_state,
            all_next_action_scores,
            logged_action_idxs,
            discount_tensor,
            not_done_mask,
        )

        if self.maxq_learning:
            possible_actions_mask = learning_input.possible_actions_mask

        if self.bcq:
            action_on_policy = get_valid_actions_from_imitator(
                self.bcq_imitator,
                learning_input.state.float_features,
                self.bcq_drop_threshold,
            )
            possible_actions_mask *= action_on_policy

        model_action_idxs = self.get_max_q_values(
            self.all_action_scores,
            possible_actions_mask if self.maxq_learning else learning_input.action,
        )[1]
        self.loss_reporter.report(
            td_loss=self.loss,
            reward_loss=reward_loss,
            logged_actions=logged_action_idxs,
            logged_propensities=training_batch.extras.action_probability,
            logged_rewards=rewards,
            logged_values=None,  # Compute at end of each epoch for CPE
            model_propensities=model_propensities,
            model_rewards=model_rewards,
            model_values=self.all_action_scores,
            model_values_on_logged_actions=None,  # Compute at end of each epoch for CPE
            model_action_idxs=model_action_idxs,
        )

    def calculate_cpes(
        self,
        training_batch,
        states,
        next_states,
        all_next_action_scores,
        logged_action_idxs,
        discount_tensor,
        not_done_mask,
    ):
        if not self.calc_cpe_in_training:
            return None, None, None

        if training_batch.extras.metrics is None:
            metrics_reward_concat_real_vals = training_batch.training_input.reward
        else:
            metrics_reward_concat_real_vals = torch.cat(
                (training_batch.training_input.reward, training_batch.extras.metrics),
                dim=1,
            )

        model_propensities_next_states = masked_softmax(
            all_next_action_scores,
            training_batch.training_input.possible_next_actions_mask
            if self.maxq_learning
            else training_batch.training_input.next_action,
            self.rl_temperature,
        )

        ######### Train separate reward network for CPE evaluation #############
        # FIXME: the reward network should be outputing a tensor, not a q-value object
        reward_estimates = self.reward_network(states).q_values
        reward_estimates_for_logged_actions = reward_estimates.gather(
            1, self.reward_idx_offsets + logged_action_idxs
        )
        reward_loss = F.mse_loss(
            reward_estimates_for_logged_actions, metrics_reward_concat_real_vals
        )
        reward_loss.backward()
        self._maybe_run_optimizer(
            self.reward_network_optimizer, self.minibatches_per_step
        )

        ######### Train separate q-network for CPE evaluation #############
        metric_q_values = self.q_network_cpe(states).q_values.gather(
            1, self.reward_idx_offsets + logged_action_idxs
        )
        all_metrics_target_q_values = torch.chunk(
            self.q_network_cpe_target(next_states).q_values.detach(),
            len(self.metrics_to_score),
            dim=1,
        )
        target_metric_q_values = []
        for i, per_metric_target_q_values in enumerate(all_metrics_target_q_values):
            per_metric_next_q_values = torch.sum(
                per_metric_target_q_values * model_propensities_next_states,
                1,
                keepdim=True,
            )
            per_metric_next_q_values = per_metric_next_q_values * not_done_mask
            per_metric_target_q_values = metrics_reward_concat_real_vals[
                :, i : i + 1
            ] + (discount_tensor * per_metric_next_q_values)
            target_metric_q_values.append(per_metric_target_q_values)

        target_metric_q_values = torch.cat(target_metric_q_values, dim=1)
        metric_q_value_loss = self.q_network_loss(
            metric_q_values, target_metric_q_values
        )
        metric_q_value_loss.backward()
        self._maybe_run_optimizer(
            self.q_network_cpe_optimizer, self.minibatches_per_step
        )

        # Use the soft update rule to update target network
        self._maybe_soft_update(
            self.q_network_cpe,
            self.q_network_cpe_target,
            self.tau,
            self.minibatches_per_step,
        )

        model_propensities = masked_softmax(
            self.all_action_scores,
            training_batch.training_input.possible_actions_mask
            if self.maxq_learning
            else training_batch.training_input.action,
            self.rl_temperature,
        )
        model_rewards = reward_estimates[
            :,
            torch.arange(
                self.reward_idx_offsets[0],
                self.reward_idx_offsets[0] + self.num_actions,
            ),
        ]
        return reward_loss, model_rewards, model_propensities

    def internal_prediction(self, input):
        """
        Only used by Gym
        """
        self.q_network.eval()
        with torch.no_grad():
            input = torch.from_numpy(np.array(input)).type(self.dtype)
            q_values = self.q_network(
                rlt.StateInput(rlt.FeatureVector(float_features=input))
            )
            q_values = q_values.q_values.cpu()
        self.q_network.train()

        if self.bcq:
            action_preds = torch.tensor(self.bcq_imitator(input.cpu()))
            action_preds /= torch.max(action_preds, dim=1)[0]
            action_off_policy = (action_preds < self.bcq_drop_threshold).float()
            action_off_policy *= self.ACTION_NOT_POSSIBLE_VAL
            q_values += action_off_policy

        return q_values.data.numpy()

    def internal_reward_estimation(self, input):
        """
        Only used by Gym
        """
        self.reward_network.eval()
        with torch.no_grad():
            input = torch.from_numpy(np.array(input)).type(self.dtype)
            reward_estimates = self.reward_network(
                rlt.StateInput(rlt.FeatureVector(float_features=input))
            )
        self.reward_network.train()
        return reward_estimates.q_values.cpu().data.numpy()
