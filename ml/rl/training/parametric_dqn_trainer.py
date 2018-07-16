#!/usr/bin/env python3

from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ml.rl.caffe_utils import StackedArray
from ml.rl.preprocessing.normalization import (
    NormalizationParameters,
    get_num_output_features,
)
from ml.rl.thrift.core.ttypes import (
    AdditionalFeatureTypes,
    ContinuousActionModelParameters,
)
from ml.rl.training.rl_trainer_pytorch import (
    DEFAULT_ADDITIONAL_FEATURE_TYPES,
    GenericFeedForwardNetwork,
    RLTrainer,
)
from ml.rl.training.evaluator import Evaluator
from ml.rl.training.parametric_dqn_predictor import ParametricDQNPredictor
from ml.rl.training.training_data_page import TrainingDataPage


class ParametricDQNTrainer(RLTrainer):
    def __init__(
        self,
        parameters: ContinuousActionModelParameters,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        action_normalization_parameters: Dict[int, NormalizationParameters],
        use_gpu=False,
        additional_feature_types: AdditionalFeatureTypes = DEFAULT_ADDITIONAL_FEATURE_TYPES,
    ) -> None:
        self.minibatch_size = parameters.training.minibatch_size
        self.state_normalization_parameters = state_normalization_parameters
        self.action_normalization_parameters = action_normalization_parameters
        self.num_features = get_num_output_features(
            state_normalization_parameters
        ) + get_num_output_features(action_normalization_parameters)

        # ensure state and action IDs have no intersection
        overlapping_features = set(state_normalization_parameters.keys()) & set(
            action_normalization_parameters.keys()
        )
        assert len(overlapping_features) == 0, (
            "There are some overlapping state and action features: "
            + str(overlapping_features)
        )

        parameters.training.layers[0] = self.num_features
        parameters.training.layers[-1] = 1

        RLTrainer.__init__(self, parameters, use_gpu, additional_feature_types)

        self.q_network = GenericFeedForwardNetwork(
            parameters.training.layers, parameters.training.activations
        )
        self.q_network_target = deepcopy(self.q_network)
        self._set_optimizer(parameters.training.optimizer)
        self.q_network_optimizer = self.optimizer_func(
            self.q_network.parameters(), lr=parameters.training.learning_rate
        )

        self.reward_network = GenericFeedForwardNetwork(
            parameters.training.layers, parameters.training.activations
        )
        self.reward_network_optimizer = self.optimizer_func(
            self.reward_network.parameters(), lr=parameters.training.learning_rate
        )

        if self.use_gpu:
            self.q_network.cuda()
            self.q_network_target.cuda()
            self.reward_network.cuda()

    def get_max_q_values(self, states, possible_actions, possible_actions_lengths):
        """
        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        :param possible_actions: Numpy array with shape (batch_size, action_dim).
            possible_next_actions[i][j] = 1 iff the agent can take action j from
            state i.
        :param possible_actions_lengths: Numpy array that describes number of
            possible_actions per item in minibatch
        """
        # TODO (edoardoc): Refactor to use EmbeddingBag as outlined in:
        # https://fb.facebook.com/groups/1405155842844877/permalink/2268098803217239/

        if isinstance(possible_actions, StackedArray):
            possible_actions = possible_actions.values

        q_network_input = []
        pna_len_cumsum = possible_actions_lengths.cumsum()

        state_idx, pa_idx_lo = 0, 0
        for pa_idx_hi in pna_len_cumsum:
            for idx in range(pa_idx_lo, pa_idx_hi):
                q_network_input.append(
                    np.concatenate([states[state_idx], possible_actions[idx]])
                )
            state_idx += 1
            pa_idx_lo = pa_idx_hi

        q_network_input = torch.from_numpy(np.array(q_network_input)).type(self.dtype)
        q_values = self.q_network_target(q_network_input).detach().cpu().data.numpy()

        max_q_values = []
        q_val_idx_lo = 0
        for q_val_idx_hi in pna_len_cumsum:
            max_q = self.ACTION_NOT_POSSIBLE_VAL
            for idx in range(q_val_idx_lo, q_val_idx_hi):
                if q_values[idx] > max_q:
                    max_q = q_values[idx][0]
            max_q_values.append(max_q)
            q_val_idx_lo = q_val_idx_hi

        return Variable(torch.from_numpy(np.array(max_q_values)).type(self.dtype))

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
                training_samples.next_states,
                training_samples.possible_next_actions,
                training_samples.possible_next_actions_lengths,
            )
        else:
            # SARSA
            next_q_values = self.get_next_action_q_values(
                training_samples.next_states, training_samples.next_actions
            )

        filtered_max_q_vals = next_q_values * not_done_mask

        if self.minibatch >= self.reward_burnin:
            target_q_values = rewards + (discount_tensor * filtered_max_q_vals)
        else:
            target_q_values = rewards

        # Get Q-value of action taken
        q_values = self.q_network(torch.cat((states, actions), dim=1)).squeeze()
        self.all_action_scores = deepcopy(q_values.detach())

        value_loss = F.mse_loss(q_values, target_q_values)
        self.loss = value_loss.detach()

        self.q_network_optimizer.zero_grad()
        value_loss.backward()
        self.q_network_optimizer.step()

        if self.minibatch >= self.reward_burnin:
            # Use the soft update rule to update target network
            self._soft_update(self.q_network, self.q_network_target, self.tau)
        else:
            # Reward burnin: force target network
            self._soft_update(self.q_network, self.q_network_target, 1.0)

        # get reward estimates
        reward_estimates = self.reward_network(
            torch.cat((states, actions), dim=1)
        ).squeeze()
        reward_loss = F.mse_loss(reward_estimates, rewards)
        self.reward_network_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_network_optimizer.step()

        # Policy evaluation logic
        if training_samples.reward_timelines is not None:
            ground_truth = np.array(
                [
                    self.get_value_from_timeline(self.gamma, rt)
                    for rt in training_samples.reward_timelines
                ]
            ).reshape(-1, 1)
        else:
            ground_truth = None

        if evaluator is not None:
            self.evaluate(
                evaluator,
                training_samples.actions,
                training_samples.propensities,
                ground_truth,
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
