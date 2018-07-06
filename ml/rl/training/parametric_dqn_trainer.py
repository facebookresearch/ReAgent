#!/usr/bin/env python3

from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ml.rl.preprocessing.normalization import (
    NormalizationParameters,
    get_num_output_features,
)
from ml.rl.thrift.core.ttypes import (
    AdditionalFeatureTypes,
    DiscreteActionModelParameters,
)
from ml.rl.training.rl_trainer_pytorch import (
    DEFAULT_ADDITIONAL_FEATURE_TYPES,
    GenericFeedForwardNetwork,
    RLTrainer,
)
from ml.rl.training.evaluator import Evaluator
from ml.rl.training.training_data_page import TrainingDataPage


class ParametricDQNTrainer(RLTrainer):
    def __init__(
        self,
        parameters: DiscreteActionModelParameters,
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

        if self.use_gpu:
            self.q_network.cuda()
            self.q_network_target.cuda()

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
        q_values = self.q_network_target(q_network_input).cpu().data.numpy()

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

        # Compute max a' Q(s', a') over all possible actions using target network
        max_q_values = self.get_max_q_values(
            training_samples.next_states,
            training_samples.possible_next_actions,
            training_samples.possible_next_actions_lengths,
        )
        filtered_max_q_vals = max_q_values * not_done_mask

        if self.minibatch >= self.reward_burnin:
            target_q_values = rewards + (discount_tensor * filtered_max_q_vals)
        else:
            target_q_values = rewards

        # Get Q-value of action taken
        q_values = self.q_network(torch.cat((states, actions), dim=1)).squeeze()

        loss = F.mse_loss(q_values, target_q_values)
        self.q_network_optimizer.zero_grad()
        loss.backward()
        self.q_network_optimizer.step()

        if self.minibatch >= self.reward_burnin:
            # Use the soft update rule to update target network
            self._soft_update(self.q_network, self.q_network_target, self.tau)
        else:
            # Reward burnin: force target network
            self._soft_update(self.q_network, self.q_network_target, 1.0)

    def evaluate(
        self,
        evaluator: Evaluator,
        td_loss: Optional[np.ndarray],
        logged_actions: Optional[np.ndarray],
        logged_propensities: Optional[np.ndarray],
        logged_rewards: Optional[np.ndarray],
        logged_values: Optional[np.ndarray],
        all_action_scores: Optional[np.ndarray],
    ):
        model_values_on_logged_actions = all_action_scores

        evaluator.report(
            td_loss,
            None,
            None,
            None,
            logged_values,
            None,
            None,
            model_values_on_logged_actions,
        )
