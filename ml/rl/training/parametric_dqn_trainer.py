#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from copy import deepcopy
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from ml.rl.models.dueling_q_network import DuelingQNetwork
from ml.rl.models.fully_connected_network import FullyConnectedNetwork
from ml.rl.preprocessing.normalization import (
    NormalizationParameters,
    get_num_output_features,
)
from ml.rl.thrift.core.ttypes import (
    AdditionalFeatureTypes,
    ContinuousActionModelParameters,
)
from ml.rl.training.dqn_trainer_base import DQNTrainerBase
from ml.rl.training.parametric_dqn_predictor import ParametricDQNPredictor
from ml.rl.training.parametric_inner_product import ParametricInnerProduct
from ml.rl.training.rl_trainer_pytorch import (
    DEFAULT_ADDITIONAL_FEATURE_TYPES,
    RLTrainer,
)
from ml.rl.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


class ParametricDQNTrainer(DQNTrainerBase):
    def __init__(
        self,
        parameters: ContinuousActionModelParameters,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        action_normalization_parameters: Dict[int, NormalizationParameters],
        use_gpu: bool = False,
        additional_feature_types: AdditionalFeatureTypes = DEFAULT_ADDITIONAL_FEATURE_TYPES,
        metrics_to_score=None,
        gradient_handler=None,
        use_all_avail_gpus: bool = False,
    ) -> None:

        self.double_q_learning = parameters.rainbow.double_q_learning
        self.warm_start_model_path = parameters.training.warm_start_model_path
        self.minibatch_size = parameters.training.minibatch_size
        self.state_normalization_parameters = state_normalization_parameters
        self.action_normalization_parameters = action_normalization_parameters
        self.num_state_features = get_num_output_features(
            state_normalization_parameters
        )
        self.num_action_features = get_num_output_features(
            action_normalization_parameters
        )
        self.num_features = self.num_state_features + self.num_action_features

        # ensure state and action IDs have no intersection
        overlapping_features = set(state_normalization_parameters.keys()) & set(
            action_normalization_parameters.keys()
        )
        assert len(overlapping_features) == 0, (
            "There are some overlapping state and action features: "
            + str(overlapping_features)
        )

        reward_network_layers = deepcopy(parameters.training.layers)
        reward_network_layers[0] = self.num_features
        reward_network_layers[-1] = 1

        if parameters.rainbow.dueling_architecture:
            parameters.training.layers[0] = self.num_state_features
            parameters.training.layers[-1] = 1
        elif parameters.training.factorization_parameters is None:
            parameters.training.layers[0] = self.num_features
            parameters.training.layers[-1] = 1
        else:
            parameters.training.factorization_parameters.state.layers[
                0
            ] = self.num_state_features
            parameters.training.factorization_parameters.action.layers[
                0
            ] = self.num_action_features

        RLTrainer.__init__(
            self,
            parameters,
            use_gpu,
            additional_feature_types,
            metrics_to_score,
            gradient_handler,
        )

        self.q_network = self._get_model(
            parameters.training, parameters.rainbow.dueling_architecture
        )

        self.q_network_target = deepcopy(self.q_network)
        self._set_optimizer(parameters.training.optimizer)
        self.q_network_optimizer = self.optimizer_func(
            self.q_network.parameters(),
            lr=parameters.training.learning_rate,
            weight_decay=parameters.training.l2_decay,
        )

        self.reward_network = FullyConnectedNetwork(
            reward_network_layers, parameters.training.activations
        )
        self.reward_network_optimizer = self.optimizer_func(
            self.reward_network.parameters(), lr=parameters.training.learning_rate
        )

        if self.use_gpu:
            self.q_network.cuda()
            self.q_network_target.cuda()
            self.reward_network.cuda()

            if use_all_avail_gpus:
                self.q_network = torch.nn.DataParallel(self.q_network)
                self.q_network_target = torch.nn.DataParallel(self.q_network_target)
                self.reward_network = torch.nn.DataParallel(self.reward_network)

    def _get_model(self, training_parameters, dueling_architecture=False):
        if dueling_architecture:
            return DuelingQNetwork(
                training_parameters.layers,
                training_parameters.activations,
                action_dim=self.num_action_features,
            )
        elif training_parameters.factorization_parameters is None:
            return FullyConnectedNetwork(
                training_parameters.layers,
                training_parameters.activations,
                use_noisy_linear_layers=training_parameters.use_noisy_linear_layers,
            )
        else:
            return ParametricInnerProduct(
                FullyConnectedNetwork(
                    training_parameters.factorization_parameters.state.layers,
                    training_parameters.factorization_parameters.state.activations,
                ),
                FullyConnectedNetwork(
                    training_parameters.factorization_parameters.action.layers,
                    training_parameters.factorization_parameters.action.activations,
                ),
                self.num_state_features,
                self.num_action_features,
            )

    def get_detached_q_values(
        self, state_action_pairs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Gets the q values from the model and target networks """
        with torch.no_grad():
            q_values = self.q_network(state_action_pairs)
            q_values_target = self.q_network_target(state_action_pairs)
        return q_values, q_values_target

    def train(self, training_samples: TrainingDataPage) -> None:
        if self.minibatch == 0:
            # Assume that the tensors are the right shape after the first minibatch
            assert (
                training_samples.states.shape[0] == self.minibatch_size
            ), "Invalid shape: " + str(training_samples.states.shape)
            assert (
                training_samples.next_states.shape == training_samples.states.shape
            ), "Invalid shape: " + str(training_samples.next_states.shape)
            assert (
                training_samples.not_terminal.shape == training_samples.rewards.shape
            ), "Invalid shape: " + str(training_samples.not_terminal.shape)

            assert (
                training_samples.actions.shape[0] == self.minibatch_size
            ), "Invalid shape: " + str(training_samples.actions.shape)
            assert (
                training_samples.possible_next_actions_mask.shape[0]
                == self.minibatch_size
            ), "Invalid shape: " + str(
                training_samples.possible_next_actions_mask.shape
            )
            assert (
                training_samples.possible_next_actions_mask.shape
                == training_samples.actions.shape
            ), (
                "Invalid shape: "
                + str(training_samples.possible_next_actions_mask.shape)
                + " != "
                + str(training_samples.actions.shape)
            )

            assert (
                training_samples.possible_next_actions_state_concat.shape[0]
                == training_samples.possible_next_actions_mask.shape[0]
                * training_samples.possible_next_actions_mask.shape[1]
            ), (
                "Invalid shape: "
                + str(training_samples.possible_next_actions_state_concat.shape)
                + " != "
                + str(training_samples.possible_next_actions_mask.shape)
            )
            assert (
                training_samples.possible_next_actions_state_concat.shape[0]
                == training_samples.next_actions.shape[0]
                * training_samples.next_actions.shape[1]
            ), (
                "Invalid shape: "
                + str(training_samples.possible_next_actions_state_concat.shape)
                + " != "
                + str(training_samples.next_actions.shape)
            )

        self.minibatch += 1

        states = training_samples.states.detach().requires_grad_(True)
        actions = training_samples.actions
        state_action_pairs = torch.cat((states, actions), dim=1)

        rewards = training_samples.rewards
        discount_tensor = torch.full(
            training_samples.time_diffs.shape, self.gamma
        ).type(self.dtype)
        not_done_mask = training_samples.not_terminal

        if self.use_seq_num_diff_as_time_diff:
            discount_tensor = discount_tensor.pow(training_samples.time_diffs)

        if self.maxq_learning:
            all_next_q_values, all_next_q_values_target = self.get_detached_q_values(
                training_samples.possible_next_actions_state_concat
            )
            # Compute max a' Q(s', a') over all possible actions using target network
            next_q_values, _ = self.get_max_q_values(
                all_next_q_values,
                all_next_q_values_target,
                training_samples.possible_next_actions_mask,
            )
        else:
            # SARSA
            next_q_values, _ = self.get_detached_q_values(
                torch.cat(
                    (training_samples.next_states, training_samples.next_actions), dim=1
                )
            )

        assert next_q_values.shape == not_done_mask.shape, (
            "Invalid shapes: "
            + str(next_q_values.shape)
            + " != "
            + str(not_done_mask.shape)
        )
        filtered_max_q_vals = next_q_values * not_done_mask

        if self.minibatch < self.reward_burnin:
            target_q_values = rewards
        else:
            assert discount_tensor.shape == filtered_max_q_vals.shape, (
                "Invalid shapes: "
                + str(discount_tensor.shape)
                + " != "
                + str(filtered_max_q_vals.shape)
            )
            target_q_values = rewards + (discount_tensor * filtered_max_q_vals)

        # Get Q-value of action taken
        q_values = self.q_network(state_action_pairs)
        all_action_scores = q_values.detach()
        self.model_values_on_logged_actions = q_values.detach()

        value_loss = self.q_network_loss(q_values, target_q_values)
        self.loss = value_loss.detach()

        self.q_network_optimizer.zero_grad()
        value_loss.backward()
        if self.gradient_handler:
            self.gradient_handler(self.q_network.parameters())
        self.q_network_optimizer.step()

        if self.minibatch < self.reward_burnin:
            # Reward burnin: force target network
            self._soft_update(self.q_network, self.q_network_target, 1.0)
        else:
            # Use the soft update rule to update target network
            self._soft_update(self.q_network, self.q_network_target, self.tau)

        # get reward estimates
        reward_estimates = self.reward_network(state_action_pairs)
        reward_loss = F.mse_loss(reward_estimates, rewards)
        self.reward_network_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_network_optimizer.step()

        self.loss_reporter.report(
            td_loss=self.loss,
            reward_loss=reward_loss,
            model_values_on_logged_actions=all_action_scores,
        )

    def predictor(self) -> ParametricDQNPredictor:
        """Builds a ParametricDQNPredictor."""
        return ParametricDQNPredictor.export(
            self,
            self.state_normalization_parameters,
            self.action_normalization_parameters,
            self._additional_feature_types.int_features,
            self.use_gpu,
        )

    def export(self) -> ParametricDQNPredictor:
        return self.predictor()
