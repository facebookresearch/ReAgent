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
from ml.rl.training.dqn_predictor import DQNPredictor
from ml.rl.training.evaluator import Evaluator
from ml.rl.training.rl_trainer_pytorch import (
    DEFAULT_ADDITIONAL_FEATURE_TYPES,
    GenericFeedForwardNetwork,
    RLTrainer,
)
from ml.rl.training.training_data_page import TrainingDataPage


class DQNTrainer(RLTrainer):
    def __init__(
        self,
        parameters: DiscreteActionModelParameters,
        state_normalization_parameters: Dict[int, NormalizationParameters],
        use_gpu=False,
        additional_feature_types: AdditionalFeatureTypes = DEFAULT_ADDITIONAL_FEATURE_TYPES,
    ) -> None:

        self.minibatch_size = parameters.training.minibatch_size
        self.maxq_learning = parameters.rl.maxq_learning
        self._actions = parameters.actions if parameters.actions is not None else []

        self.reward_shape = {}  # type: Dict[int, float]
        if parameters.rl.reward_boost is not None and self._actions is not None:
            for k in parameters.rl.reward_boost.keys():
                i = self._actions.index(k)
                self.reward_shape[i] = parameters.rl.reward_boost[k]

        if parameters.training.cnn_parameters is None:
            self.state_normalization_parameters: Optional[
                Dict[int, NormalizationParameters]
            ] = state_normalization_parameters
            self.num_features = get_num_output_features(state_normalization_parameters)
            parameters.training.layers[0] = self.num_features
        else:
            self.state_normalization_parameters = None
        parameters.training.layers[-1] = self.num_actions

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

    @property
    def num_actions(self) -> int:
        return len(self._actions)

    def get_max_q_values(self, states, possible_actions):
        """
        Used in Q-learning update.
        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        :param possible_actions: Numpy array with shape (batch_size, action_dim).
            possible_next_actions[i][j] = 1 iff the agent can take action j from
            state i.
        """
        q_values = self.q_network_target(states).detach()
        self.all_action_scores = deepcopy(q_values)

        # Set q-values of impossible actions to a very large negative number.
        inverse_pna = 1 - possible_actions
        impossible_action_penalty = self.ACTION_NOT_POSSIBLE_VAL * inverse_pna
        q_values += impossible_action_penalty
        return Variable(torch.max(q_values, 1)[0])

    def get_next_action_q_values(self, states, next_actions):
        """
        Used in SARSA update.
        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        :param next_actions: Numpy array with shape (batch_size, action_dim).
        """
        q_values = self.q_network_target(states).detach()
        self.all_action_scores = deepcopy(q_values)
        return Variable(torch.sum(q_values * next_actions, 1))

    def train(
        self, training_samples: TrainingDataPage, evaluator: Optional[Evaluator] = None
    ) -> None:

        # Apply reward boost if specified
        if len(self.reward_shape) > 0:
            boost_idxs = np.argmax(training_samples.actions, 1)
            boosts = np.array([self.reward_shape[x] for x in boost_idxs])
            training_samples.rewards += boosts

        self.minibatch += 1
        states = Variable(torch.from_numpy(training_samples.states).type(self.dtype))
        actions = Variable(torch.from_numpy(training_samples.actions).type(self.dtype))
        rewards = Variable(torch.from_numpy(training_samples.rewards).type(self.dtype))
        next_states = Variable(
            torch.from_numpy(training_samples.next_states).type(self.dtype)
        )
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
        if self.maxq_learning:
            possible_next_actions = Variable(
                torch.from_numpy(training_samples.possible_next_actions).type(
                    self.dtype
                )
            )
            next_q_values = self.get_max_q_values(next_states, possible_next_actions)
        # SARSA
        else:
            next_actions = Variable(
                torch.from_numpy(training_samples.next_actions).type(self.dtype)
            )
            next_q_values = self.get_next_action_q_values(next_states, next_actions)

        filtered_next_q_vals = next_q_values * not_done_mask

        if self.minibatch >= self.reward_burnin:
            target_q_values = rewards + (discount_tensor * filtered_next_q_vals)
        else:
            target_q_values = rewards

        # Get Q-value of action taken
        q_values = torch.sum(self.q_network(states) * actions, 1)
        loss = F.mse_loss(q_values, target_q_values)
        self.loss = loss.detach()

        self.q_network_optimizer.zero_grad()
        loss.backward()
        self.q_network_optimizer.step()

        if self.minibatch >= self.reward_burnin:
            # Use the soft update rule to update target network
            self._soft_update(self.q_network, self.q_network_target, self.tau)
        else:
            # Reward burnin: force target network
            self._soft_update(self.q_network, self.q_network_target, 1.0)

        # get reward estimates
        reward_estimates = (
            self.reward_network(states).gather(
                1, actions.argmax(1).unsqueeze(1)
            ).squeeze()
        )
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
                np.expand_dims(training_samples.rewards, axis=1),
                ground_truth,
            )


    def evaluate(
        self,
        evaluator: Evaluator,
        logged_actions: Optional[np.ndarray],
        logged_propensities: Optional[np.ndarray],
        logged_rewards: Optional[np.ndarray],
        logged_values: Optional[np.ndarray],
    ):
        model_propensities, model_values_on_logged_actions, maxq_action_idxs = (
            None,
            None,
            None,
        )
        if self.all_action_scores is not None:
            self.all_action_scores = self.all_action_scores.cpu().numpy()
            model_propensities = Evaluator.softmax(
                self.all_action_scores, self.rl_temperature
            )
            maxq_action_idxs = self.all_action_scores.argmax(axis=1)
            if logged_actions is not None:
                model_values_on_logged_actions = np.sum(
                    (logged_actions * self.all_action_scores), axis=1, keepdims=True
                )

        evaluator.report(
            self.loss.cpu().numpy(),
            logged_actions,
            logged_propensities,
            logged_rewards,
            logged_values,
            model_propensities,
            self.all_action_scores,
            model_values_on_logged_actions,
            maxq_action_idxs,
        )

    def predictor(self) -> DQNPredictor:
        """Builds a DQNPredictor."""
        return DQNPredictor.export(
            self,
            self._actions,
            self.state_normalization_parameters,
            self._additional_feature_types.int_features,
            self.use_gpu,
        )
