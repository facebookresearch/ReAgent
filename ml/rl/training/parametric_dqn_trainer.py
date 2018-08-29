#!/usr/bin/env python3

from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from ml.rl.caffe_utils import arange_expand
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
    DuelingArchitectureQNetwork,
    GenericFeedForwardNetwork,
    RLTrainer,
)
from ml.rl.training.training_data_page import TrainingDataPage
from torch.autograd import Variable


class ParametricDQNTrainer(RLTrainer):
    def __init__(
        self,
        parameters: ContinuousActionModelParameters,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        action_normalization_parameters: Dict[int, NormalizationParameters],
        use_gpu=False,
        additional_feature_types: AdditionalFeatureTypes = DEFAULT_ADDITIONAL_FEATURE_TYPES,
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

        RLTrainer.__init__(self, parameters, use_gpu, additional_feature_types, None)

        self.q_network = self._get_model(
            parameters.training, parameters.rainbow.dueling_architecture
        )

        self.q_network_target = deepcopy(self.q_network)
        self._set_optimizer(parameters.training.optimizer)
        self.q_network_optimizer = self.optimizer_func(
            self.q_network.parameters(), lr=parameters.training.learning_rate
        )

        self.reward_network = GenericFeedForwardNetwork(
            reward_network_layers, parameters.training.activations
        )
        self.reward_network_optimizer = self.optimizer_func(
            self.reward_network.parameters(), lr=parameters.training.learning_rate
        )

        if self.use_gpu:
            self.q_network.cuda()
            self.q_network_target.cuda()
            self.reward_network.cuda()

    def _get_model(self, training_parameters, dueling_architecture=False):
        if dueling_architecture:
            return DuelingArchitectureQNetwork(
                training_parameters.layers,
                training_parameters.activations,
                action_dim=self.num_action_features,
            )
        elif training_parameters.factorization_parameters is None:
            return GenericFeedForwardNetwork(
                training_parameters.layers, training_parameters.activations
            )
        else:
            return ParametricInnerProduct(
                GenericFeedForwardNetwork(
                    training_parameters.factorization_parameters.state.layers,
                    training_parameters.factorization_parameters.state.activations,
                ),
                GenericFeedForwardNetwork(
                    training_parameters.factorization_parameters.action.layers,
                    training_parameters.factorization_parameters.action.activations,
                ),
                self.num_state_features,
                self.num_action_features,
            )

    def calculate_q_values(self, state_pas_concats, pas_lens):
        row_nums = np.arange(len(pas_lens))
        row_idxs = np.repeat(row_nums, pas_lens)
        col_idxs = arange_expand(pas_lens)

        dense_idxs = torch.LongTensor((row_idxs, col_idxs)).type(self.dtypelong)
        q_network_input = torch.from_numpy(state_pas_concats).type(self.dtype)

        q_values = self.q_network(q_network_input).detach().squeeze()

        dense_dim = [len(pas_lens), max(pas_lens)]
        # Add specific fingerprint to q-values so that after sparse -> dense we can
        # subtract the fingerprint to identify the 0's added in sparse -> dense
        q_values.add_(self.FINGERPRINT)
        sparse_q = torch.sparse_coo_tensor(dense_idxs, q_values, dense_dim)
        dense_q = sparse_q.to_dense()
        dense_q.add_(self.FINGERPRINT * -1)
        dense_q[dense_q == self.FINGERPRINT * -1] = self.ACTION_NOT_POSSIBLE_VAL

        return dense_q.cpu().numpy()

    def get_max_q_values(self, next_state_pnas_concat, pnas_lens, double_q_learning):
        """
        :param next_state_pnas_concat: Numpy array with shape
            (sum(pnas_lens), state_dim + action_dim). Each row
            contains a representation of a state + possible next action pair.
        :param pnas_lens: Numpy array that describes number of
            possible_actions per item in minibatch
        :param double_q_learning: bool to use double q-learning
        """
        row_nums = np.arange(len(pnas_lens))
        row_idxs = np.repeat(row_nums, pnas_lens)
        col_idxs = arange_expand(pnas_lens)

        dense_idxs = torch.LongTensor((row_idxs, col_idxs)).type(self.dtypelong)
        q_network_input = torch.from_numpy(next_state_pnas_concat).type(self.dtype)

        if double_q_learning:
            q_values = self.q_network(q_network_input).detach().squeeze()
            q_values_target = self.q_network_target(q_network_input).detach().squeeze()
        else:
            q_values = self.q_network_target(q_network_input).detach().squeeze()

        dense_dim = [len(pnas_lens), max(pnas_lens)]
        # Add specific fingerprint to q-values so that after sparse -> dense we can
        # subtract the fingerprint to identify the 0's added in sparse -> dense
        q_values.add_(self.FINGERPRINT)
        sparse_q = torch.sparse_coo_tensor(dense_idxs, q_values, dense_dim)
        dense_q = sparse_q.to_dense()
        dense_q.add_(self.FINGERPRINT * -1)
        dense_q[dense_q == self.FINGERPRINT * -1] = self.ACTION_NOT_POSSIBLE_VAL
        max_q_values, max_indexes = torch.max(dense_q, dim=1)

        if double_q_learning:
            sparse_q_target = torch.sparse_coo_tensor(
                dense_idxs, q_values_target, dense_dim
            )
            dense_q_values_target = sparse_q_target.to_dense()
            max_q_values = torch.gather(
                dense_q_values_target, 1, max_indexes.unsqueeze(1)
            )

        return Variable(max_q_values.squeeze())

    def get_next_action_q_values(self, states, next_actions):
        """
        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        :param next_actions: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of an action.
        """
        q_network_input = np.concatenate([states, next_actions], 1)
        q_network_input = torch.from_numpy(q_network_input).type(self.dtype)
        return Variable(self.q_network_target(q_network_input).detach().squeeze())

    def train(
        self, training_samples: TrainingDataPage, evaluator=None, episode_values=None
    ) -> None:

        self.minibatch += 1
        states = Variable(torch.from_numpy(training_samples.states).type(self.dtype))
        actions = Variable(torch.from_numpy(training_samples.actions).type(self.dtype))
        state_action_pairs = torch.cat((states, actions), dim=1)
        rewards = Variable(torch.from_numpy(training_samples.rewards).type(self.dtype))
        time_diffs = torch.tensor(training_samples.time_diffs).type(self.dtype)
        discount_tensor = torch.tensor(np.full(len(rewards), self.gamma)).type(
            self.dtype
        )
        not_done_mask = Variable(
            torch.from_numpy(training_samples.not_terminals.astype(int))
        ).type(self.dtype)

        if self.use_seq_num_diff_as_time_diff:
            discount_tensor = discount_tensor.pow(time_diffs)

        if self.maxq_learning:
            # Compute max a' Q(s', a') over all possible actions using target network
            next_q_values = self.get_max_q_values(
                training_samples.next_state_pnas_concat,
                training_samples.possible_next_actions_lengths,
                self.double_q_learning,
            )
        else:
            # SARSA
            next_q_values = self.get_next_action_q_values(
                training_samples.next_states, training_samples.next_actions
            )

        filtered_max_q_vals = next_q_values * not_done_mask

        if self.use_reward_burnin and self.minibatch < self.reward_burnin:
            target_q_values = rewards
        else:
            target_q_values = rewards + (discount_tensor * filtered_max_q_vals)

        # Get Q-value of action taken
        q_values = self.q_network(state_action_pairs)
        self.all_action_scores = deepcopy(q_values.detach())

        value_loss = self.q_network_loss(q_values.squeeze(), target_q_values)
        self.loss = value_loss.detach()

        self.q_network_optimizer.zero_grad()
        value_loss.backward()
        self.q_network_optimizer.step()

        if self.use_reward_burnin and self.minibatch < self.reward_burnin:
            # Reward burnin: force target network
            self._soft_update(self.q_network, self.q_network_target, 1.0)
        else:
            # Use the soft update rule to update target network
            self._soft_update(self.q_network, self.q_network_target, self.tau)

        # get reward estimates
        reward_estimates = self.reward_network(state_action_pairs).squeeze()
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
        logged_actions: Optional[np.ndarray],
        logged_propensities: Optional[np.ndarray],
        logged_values: Optional[np.ndarray],
    ):
        evaluator.report(
            self.loss.cpu().numpy(),
            None,
            None,
            None,
            logged_values,
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
