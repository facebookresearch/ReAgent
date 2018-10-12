#!/usr/bin/env python3

import logging
from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from ml.rl.caffe_utils import arange_expand
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
from ml.rl.training.evaluator import Evaluator
from ml.rl.training.parametric_dqn_predictor import ParametricDQNPredictor
from ml.rl.training.parametric_inner_product import ParametricInnerProduct
from ml.rl.training.rl_trainer_pytorch import (
    DEFAULT_ADDITIONAL_FEATURE_TYPES,
    RLTrainer,
)
from ml.rl.training.training_data_page import TrainingDataPage


logger = logging.getLogger(__name__)


class ParametricDQNTrainer(RLTrainer):
    def __init__(
        self,
        parameters: ContinuousActionModelParameters,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        action_normalization_parameters: Dict[int, NormalizationParameters],
        use_gpu: bool = False,
        additional_feature_types: AdditionalFeatureTypes = DEFAULT_ADDITIONAL_FEATURE_TYPES,
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
            self, parameters, use_gpu, additional_feature_types, gradient_handler
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
                training_parameters.layers, training_parameters.activations
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

    def calculate_q_values(self, state_pas_concats, pas_lens):
        row_nums = np.arange(pas_lens.shape[0], dtype=np.int64)
        row_idxs = np.repeat(row_nums, pas_lens.cpu().numpy())
        col_idxs = arange_expand(pas_lens)

        dense_idxs = torch.LongTensor((row_idxs, col_idxs)).type(self.dtypelong)

        q_values = self.q_network(state_pas_concats).squeeze().detach()

        dense_dim = [len(pas_lens), int(torch.max(pas_lens))]
        # Add specific fingerprint to q-values so that after sparse -> dense we can
        # subtract the fingerprint to identify the 0's added in sparse -> dense
        q_values.add_(self.FINGERPRINT)
        sparse_q = torch.sparse_coo_tensor(dense_idxs, q_values, size=dense_dim)
        dense_q = sparse_q.to_dense()
        dense_q.add_(self.FINGERPRINT * -1)
        dense_q[dense_q == self.FINGERPRINT * -1] = self.ACTION_NOT_POSSIBLE_VAL

        return dense_q

    def get_max_q_values(
        self, possible_next_actions_state_concat, pnas_lens, double_q_learning
    ):
        """
        :param possible_next_actions_state_concat: Numpy array with shape
            (sum(pnas_lens), state_dim + action_dim). Each row
            contains a representation of a state + possible next action pair.
        :param pnas_lens: Numpy array that describes number of
            possible_actions per item in minibatch
        :param double_q_learning: bool to use double q-learning
        """
        if isinstance(possible_next_actions_state_concat, torch.Tensor):
            q_network_input = possible_next_actions_state_concat
        else:
            q_network_input = torch.from_numpy(possible_next_actions_state_concat).type(
                self.dtype
            )

        if double_q_learning:
            q_values = self.q_network(q_network_input).squeeze().detach()
            q_values_target = self.q_network_target(q_network_input).squeeze().detach()
        else:
            q_values = self.q_network_target(q_network_input).squeeze().detach()

        max_q_values = torch.zeros(pnas_lens.size()).type(self.dtype)
        offset = torch.tensor([0]).type(self.dtypelong)
        for i, l in enumerate(pnas_lens):
            if l > 0:
                max_q_values[i], max_index = torch.max(
                    q_values[offset : offset + l], dim=0
                )
                if double_q_learning:
                    max_q_values[i] = q_values_target[offset : offset + l][max_index]
            offset += l
        return max_q_values.squeeze()

    def get_next_action_q_values(self, state_action_pairs):
        return self.q_network_target(state_action_pairs)

    def train(
        self, training_samples: TrainingDataPage, evaluator=None, episode_values=None
    ) -> None:
        if self.minibatch == 0:
            # Assume that the tensors are the right shape after the first minibatch
            assert (
                training_samples.states.shape[0] == self.minibatch_size
            ), "Invalid shape: " + str(training_samples.states.shape)
            assert (
                training_samples.actions.shape[0] == self.minibatch_size
            ), "Invalid shape: " + str(training_samples.actions.shape)
            assert training_samples.rewards.shape == torch.Size(
                [self.minibatch_size, 1]
            ), "Invalid shape: " + str(training_samples.rewards.shape)
            assert (
                training_samples.episode_values is None
                or training_samples.episode_values.shape
                == training_samples.rewards.shape
            ), "Invalid shape: " + str(training_samples.episode_values.shape)
            assert (
                training_samples.next_states.shape == training_samples.states.shape
            ), "Invalid shape: " + str(training_samples.next_states.shape)
            assert (
                training_samples.not_terminals.shape == training_samples.rewards.shape
            ), "Invalid shape: " + str(training_samples.not_terminals.shape)
            assert training_samples.possible_next_actions_state_concat.shape[1] == (
                training_samples.states.shape[1] + training_samples.actions.shape[1]
            ), (
                "Invalid shape: "
                + str(training_samples.possible_next_actions_state_concat.shape)
            )
            assert training_samples.possible_next_actions_lengths.shape == torch.Size(
                [self.minibatch_size]
            ), (
                "Invalid shape: "
                + str(training_samples.possible_next_actions_lengths.shape)
            )

        self.minibatch += 1

        states = training_samples.states.detach().requires_grad_(True)
        actions = training_samples.actions
        state_action_pairs = torch.cat((states, actions), dim=1)

        rewards = training_samples.rewards
        discount_tensor = torch.full(
            training_samples.time_diffs.shape, self.gamma
        ).type(self.dtype)
        not_done_mask = training_samples.not_terminals

        if self.use_seq_num_diff_as_time_diff:
            discount_tensor = discount_tensor.pow(training_samples.time_diffs)

        if self.maxq_learning:
            # Compute max a' Q(s', a') over all possible actions using target network
            next_q_values = self.get_max_q_values(
                training_samples.possible_next_actions_state_concat,
                training_samples.possible_next_actions_lengths,
                self.double_q_learning,
            )
        else:
            # SARSA
            next_state_action_pairs = torch.cat(
                (training_samples.next_states, training_samples.next_actions), dim=1
            )
            next_q_values = self.get_next_action_q_values(next_state_action_pairs)

        filtered_max_q_vals = next_q_values.reshape(-1, 1) * not_done_mask

        if self.minibatch < self.reward_burnin:
            target_q_values = rewards
        else:
            target_q_values = rewards + (discount_tensor * filtered_max_q_vals)

        # Get Q-value of action taken
        q_values = self.q_network(state_action_pairs)
        self.all_action_scores = q_values.detach()

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

        if evaluator is not None:
            self.evaluate(
                evaluator,
                training_samples.actions,
                training_samples.propensities,
                training_samples.episode_values,
            )

    def evaluate(
        self,
        evaluator: Evaluator,
        logged_actions: Optional[torch.Tensor],
        logged_propensities: Optional[torch.Tensor],
        logged_values: Optional[torch.Tensor],
    ):
        evaluator.report(
            self.loss.cpu().numpy(),
            None,
            None,
            None,
            logged_values.cpu().numpy() if logged_values is not None else None,
            None,
            None,
            None,
            self.all_action_scores.cpu().numpy(),
            None,
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
