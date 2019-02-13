#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from ml.rl.caffe_utils import masked_softmax, softmax
from ml.rl.models.convolutional_network import ConvolutionalNetwork
from ml.rl.models.dueling_q_network import DuelingQNetwork
from ml.rl.models.fully_connected_network import FullyConnectedNetwork
from ml.rl.preprocessing.normalization import (
    NormalizationParameters,
    get_num_output_features,
)
from ml.rl.thrift.core.ttypes import DiscreteActionModelParameters
from ml.rl.training.dqn_predictor import DQNPredictor
from ml.rl.training.dqn_trainer_base import DQNTrainerBase
from ml.rl.training.rl_trainer_pytorch import RLTrainer
from ml.rl.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


class DQNTrainer(DQNTrainerBase):
    def __init__(
        self,
        parameters: DiscreteActionModelParameters,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        use_gpu: bool = False,
        metrics_to_score=None,
        gradient_handler=None,
        use_all_avail_gpus: bool = False,
    ) -> None:

        self.double_q_learning = parameters.rainbow.double_q_learning
        self.bcq = parameters.rainbow.bcq
        self.bcq_drop_threshold = parameters.rainbow.bcq_drop_threshold
        self.warm_start_model_path = parameters.training.warm_start_model_path
        self.minibatch_size = parameters.training.minibatch_size
        self._actions = parameters.actions if parameters.actions is not None else []

        if parameters.training.cnn_parameters is None:
            self.state_normalization_parameters: Optional[
                Dict[int, NormalizationParameters]
            ] = state_normalization_parameters
            self.num_features = get_num_output_features(state_normalization_parameters)
            logger.info("Number of state features: " + str(self.num_features))
            parameters.training.layers[0] = self.num_features
        else:
            self.state_normalization_parameters = None
        parameters.training.layers[-1] = self.num_actions

        RLTrainer.__init__(
            self,
            parameters,
            use_gpu,
            metrics_to_score,
            gradient_handler,
            actions=self._actions,
        )

        self.reward_boosts = torch.zeros([1, len(self._actions)]).type(self.dtype)
        if parameters.rl.reward_boost is not None:
            for k in parameters.rl.reward_boost.keys():
                i = self._actions.index(k)
                self.reward_boosts[0, i] = parameters.rl.reward_boost[k]

        if parameters.rainbow.dueling_architecture:
            self.q_network = DuelingQNetwork(
                parameters.training.layers,
                parameters.training.activations,
                use_batch_norm=parameters.training.use_batch_norm,
            )
        else:
            if parameters.training.cnn_parameters is None:
                self.q_network = FullyConnectedNetwork(
                    parameters.training.layers,
                    parameters.training.activations,
                    use_noisy_linear_layers=parameters.training.use_noisy_linear_layers,
                    min_std=parameters.training.weight_init_min_std,
                    use_batch_norm=parameters.training.use_batch_norm,
                )
            else:
                self.q_network = ConvolutionalNetwork(
                    parameters.training.cnn_parameters,
                    parameters.training.layers,
                    parameters.training.activations,
                    use_noisy_linear_layers=parameters.training.use_noisy_linear_layers,
                    min_std=parameters.training.weight_init_min_std,
                    use_batch_norm=parameters.training.use_batch_norm,
                )

        self.q_network_target = deepcopy(self.q_network)
        self.q_network._name = "training"
        self.q_network_target._name = "target"
        self._set_optimizer(parameters.training.optimizer)
        self.q_network_optimizer = self.optimizer_func(
            self.q_network.parameters(),
            lr=parameters.training.learning_rate,
            weight_decay=parameters.training.l2_decay,
        )
        self.clip_grad_norm = parameters.training.clip_grad_norm

        self._init_cpe_networks(parameters, use_all_avail_gpus)
        self._init_bcq_network(parameters, use_all_avail_gpus)

        if self.use_gpu:
            self.q_network.cuda()
            self.q_network_target.cuda()

            if use_all_avail_gpus:
                self.q_network = torch.nn.DataParallel(self.q_network)
                self.q_network_target = torch.nn.DataParallel(self.q_network_target)

    def _init_bcq_network(self, parameters, use_all_avail_gpus):
        # Batch constrained q-learning
        if not parameters.rainbow.bcq:
            return

        self.bcq_imitator = FullyConnectedNetwork(
            parameters.training.layers,
            parameters.training.activations,
            min_std=parameters.training.weight_init_min_std,
            use_batch_norm=parameters.training.use_batch_norm,
        )
        self.bcq_imitator_optimizer = self.optimizer_func(
            self.bcq_imitator.parameters(),
            lr=parameters.training.learning_rate,
            weight_decay=parameters.training.l2_decay,
        )

        if self.use_gpu:
            self.bcq_imitator.cuda()

    def _init_cpe_networks(self, parameters, use_all_avail_gpus):
        if not self.calc_cpe_in_training:
            return

        reward_network_layers = deepcopy(parameters.training.layers)
        if self.metrics_to_score:
            num_output_nodes = len(self.metrics_to_score) * self.num_actions
        else:
            num_output_nodes = self.num_actions

        reward_network_layers[-1] = num_output_nodes
        self.reward_idx_offsets = torch.arange(
            0, num_output_nodes, self.num_actions
        ).type(self.dtypelong)
        logger.info(
            "Reward network for CPE will have {} output nodes.".format(num_output_nodes)
        )

        if parameters.training.cnn_parameters is None:
            self.reward_network = FullyConnectedNetwork(
                reward_network_layers, parameters.training.activations
            )
            self.q_network_cpe = FullyConnectedNetwork(
                reward_network_layers, parameters.training.activations
            )
        else:
            self.reward_network = ConvolutionalNetwork(
                parameters.training.cnn_parameters,
                reward_network_layers,
                parameters.training.activations,
            )
            self.q_network_cpe = ConvolutionalNetwork(
                parameters.training.cnn_parameters,
                reward_network_layers,
                parameters.training.activations,
            )
        self.q_network_cpe_target = deepcopy(self.q_network_cpe)
        self.q_network_cpe_optimizer = self.optimizer_func(
            self.q_network_cpe.parameters(), lr=parameters.training.learning_rate
        )
        self.reward_network_optimizer = self.optimizer_func(
            self.reward_network.parameters(), lr=parameters.training.learning_rate
        )
        if self.use_gpu:
            self.reward_network.cuda()
            self.q_network_cpe.cuda()
            self.q_network_cpe_target.cuda()
            if use_all_avail_gpus:
                self.reward_network = torch.nn.DataParallel(self.reward_network)
                self.q_network_cpe = torch.nn.DataParallel(self.q_network_cpe)
                self.q_network_cpe_target = torch.nn.DataParallel(
                    self.q_network_cpe_target
                )

    @property
    def num_actions(self) -> int:
        return len(self._actions)

    def calculate_q_values(self, states):
        return self.q_network(states).detach()

    def calculate_metric_q_values(self, states):
        return self.q_network_cpe(states).detach()

    def get_detached_q_values(self, states) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            q_values = self.q_network(states)
            q_values_target = self.q_network_target(states)
        return q_values, q_values_target

    def get_next_action_q_values(self, states, next_actions):
        """
        Used in SARSA update.
        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        :param next_actions: Numpy array with shape (batch_size, action_dim).
        """
        q_values = self.q_network_target(states).detach()
        # Max-q action indexes used in CPE
        max_q_values, max_indicies = torch.max(q_values, dim=1, keepdim=True)
        return (torch.sum(q_values * next_actions, dim=1, keepdim=True), max_indicies)

    def train(self, training_samples: TrainingDataPage):

        if self.minibatch == 0:
            # Assume that the tensors are the right shape after the first minibatch
            assert (
                training_samples.states.shape[0] == self.minibatch_size
            ), "Invalid shape: " + str(training_samples.states.shape)
            assert training_samples.actions.shape == torch.Size(
                [self.minibatch_size, len(self._actions)]
            ), "Invalid shape: " + str(training_samples.actions.shape)
            assert training_samples.rewards.shape == torch.Size(
                [self.minibatch_size, 1]
            ), "Invalid shape: " + str(training_samples.rewards.shape)
            assert (
                training_samples.next_states.shape == training_samples.states.shape
            ), "Invalid shape: " + str(training_samples.next_states.shape)
            assert (
                training_samples.not_terminal.shape == training_samples.rewards.shape
            ), "Invalid shape: " + str(training_samples.not_terminal.shape)
            if training_samples.possible_next_actions_mask is not None:
                assert (
                    training_samples.possible_next_actions_mask.shape
                    == training_samples.actions.shape
                ), (
                    "Invalid shape: "
                    + str(training_samples.possible_next_actions_mask.shape)
                )
            if training_samples.propensities is not None:
                assert (
                    training_samples.propensities.shape
                    == training_samples.rewards.shape
                ), "Invalid shape: " + str(training_samples.propensities.shape)
            if training_samples.metrics is not None:
                assert (
                    training_samples.metrics.shape[0] == self.minibatch_size
                ), "Invalid shape: " + str(training_samples.metrics.shape)

        boosted_rewards = self.boost_rewards(
            training_samples.rewards, training_samples.actions
        )

        self.minibatch += 1
        states = training_samples.states.detach().requires_grad_(True)
        actions = training_samples.actions
        rewards = boosted_rewards
        discount_tensor = torch.full(
            training_samples.time_diffs.shape, self.gamma
        ).type(self.dtype)
        not_done_mask = training_samples.not_terminal

        if self.use_seq_num_diff_as_time_diff:
            time_diff = training_samples.time_diffs / self.time_diff_unit_length
            discount_tensor = discount_tensor.pow(time_diff)

        all_next_q_values, all_next_q_values_target = self.get_detached_q_values(
            training_samples.next_states
        )

        if self.bcq:
            # Batch constrained q-learning
            on_policy_actions = self.bcq_imitator(training_samples.next_states)
            on_policy_action_probs = softmax(on_policy_actions, temperature=1)
            filter_values = (
                on_policy_action_probs
                / on_policy_action_probs.max(keepdim=True, dim=1)[0]
            )
            action_on_policy = (filter_values >= self.bcq_drop_threshold).float()
            training_samples.possible_next_actions_mask *= action_on_policy
        if self.maxq_learning:
            # Compute max a' Q(s', a') over all possible actions using target network
            next_q_values, max_q_action_idxs = self.get_max_q_values_with_target(
                all_next_q_values,
                all_next_q_values_target,
                training_samples.possible_next_actions_mask,
            )
        else:
            # SARSA
            next_q_values, max_q_action_idxs = self.get_max_q_values_with_target(
                all_next_q_values,
                all_next_q_values_target,
                training_samples.next_actions,
            )

        filtered_next_q_vals = next_q_values * not_done_mask

        if self.minibatch < self.reward_burnin:
            target_q_values = rewards
        else:
            target_q_values = rewards + (discount_tensor * filtered_next_q_vals)

        # Get Q-value of action taken
        all_q_values = self.q_network(states)
        self.all_action_scores = all_q_values.detach()
        q_values = torch.sum(all_q_values * actions, 1, keepdim=True)

        loss = self.q_network_loss(q_values, target_q_values)
        self.loss = loss.detach()

        self.q_network_optimizer.zero_grad()
        loss.backward()
        if self.gradient_handler:
            self.gradient_handler(self.q_network.parameters())
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), self.clip_grad_norm
            )
        self.q_network_optimizer.step()

        if self.minibatch < self.reward_burnin:
            # Reward burnin: force target network
            self._soft_update(self.q_network, self.q_network_target, 1.0)
        else:
            # Use the soft update rule to update target network
            self._soft_update(self.q_network, self.q_network_target, self.tau)

        bcq_loss = None
        if self.bcq:
            # Batch constrained q-learning
            action_preds = self.bcq_imitator(states)
            imitator_loss = torch.nn.CrossEntropyLoss()
            # Classification label is index of action with value 1
            bcq_loss = imitator_loss(action_preds, torch.max(actions, dim=1)[1])
            self.bcq_imitator_optimizer.zero_grad()
            bcq_loss.backward()
            self.bcq_imitator_optimizer.step()

        logged_action_idxs = actions.argmax(dim=1, keepdim=True)
        reward_loss, model_rewards, model_propensities = self.calculate_cpes(
            training_samples,
            states,
            logged_action_idxs,
            max_q_action_idxs,
            discount_tensor,
            not_done_mask,
        )

        self.loss_reporter.report(
            td_loss=self.loss,
            imitator_loss=bcq_loss,
            reward_loss=reward_loss,
            logged_actions=logged_action_idxs,
            logged_propensities=training_samples.propensities,
            logged_rewards=rewards,
            logged_values=None,  # Compute at end of each epoch for CPE
            model_propensities=model_propensities,
            model_rewards=model_rewards,
            model_values=self.all_action_scores,
            model_values_on_logged_actions=None,  # Compute at end of each epoch for CPE
            model_action_idxs=self.get_max_q_values(
                self.all_action_scores, training_samples.possible_actions_mask
            )[1],
        )

    def calculate_cpes(
        self,
        training_samples,
        states,
        logged_action_idxs,
        max_q_action_idxs,
        discount_tensor,
        not_done_mask,
    ):
        if not self.calc_cpe_in_training:
            return None, None, None

        if training_samples.metrics is None:
            metrics_reward_concat_real_vals = training_samples.rewards
        else:
            metrics_reward_concat_real_vals = torch.cat(
                (training_samples.rewards, training_samples.metrics), dim=1
            )

        ######### Train separate reward network for CPE evaluation #############
        reward_estimates = self.reward_network(states)
        reward_estimates_for_logged_actions = reward_estimates.gather(
            1, self.reward_idx_offsets + logged_action_idxs
        )
        reward_loss = F.mse_loss(
            reward_estimates_for_logged_actions, metrics_reward_concat_real_vals
        )
        self.reward_network_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_network_optimizer.step()

        ######### Train separate q-network for CPE evaluation #############
        metric_q_values = self.q_network_cpe(states).gather(
            1, self.reward_idx_offsets + logged_action_idxs
        )
        metric_target_q_values = self.q_network_cpe_target(states).detach()
        max_q_values_metrics = metric_target_q_values.gather(
            1, self.reward_idx_offsets + max_q_action_idxs
        )
        filtered_max_q_values_metrics = max_q_values_metrics * not_done_mask
        if self.minibatch < self.reward_burnin:
            target_metric_q_values = metrics_reward_concat_real_vals
        else:
            target_metric_q_values = metrics_reward_concat_real_vals + (
                discount_tensor * filtered_max_q_values_metrics
            )
        metric_q_value_loss = self.q_network_loss(
            metric_q_values, target_metric_q_values
        )
        self.q_network_cpe.zero_grad()
        metric_q_value_loss.backward()
        self.q_network_cpe_optimizer.step()

        if self.minibatch < self.reward_burnin:
            # Reward burnin: force target network
            self._soft_update(self.q_network_cpe, self.q_network_cpe_target, 1.0)
        else:
            # Use the soft update rule to update target network
            self._soft_update(self.q_network_cpe, self.q_network_cpe_target, self.tau)

        model_propensities = masked_softmax(
            self.all_action_scores,
            training_samples.possible_actions_mask,
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

    def boost_rewards(
        self, rewards: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        # Apply reward boost if specified
        reward_boosts = torch.sum(
            actions.float() * self.reward_boosts, dim=1, keepdim=True
        )
        return rewards + reward_boosts

    def predictor(self, set_missing_value_to_zero=False) -> DQNPredictor:
        """Builds a DQNPredictor."""
        return DQNPredictor.export(
            self,
            self._actions,
            self.state_normalization_parameters,
            self.use_gpu,
            set_missing_value_to_zero=set_missing_value_to_zero,
        )

    def export(self) -> DQNPredictor:
        return self.predictor()
