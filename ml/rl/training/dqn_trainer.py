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


class DQNTrainer(RLTrainer):
    # Set to a very large negative number.  Guaranteed to be worse than any
    #     legitimate action
    ACTION_NOT_POSSIBLE_VAL = -1e9

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
            num_features = get_num_output_features(state_normalization_parameters)
            parameters.training.layers[0] = num_features
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

        if self.use_gpu:
            self.q_network.cuda()
            self.q_network_target.cuda()

    @property
    def num_actions(self) -> int:
        return len(self._actions)

    def get_max_q_values(self, states, possible_actions):
        """
        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        :param possible_actions: Numpy array with shape (batch_size, action_dim).
            possible_next_actions[i][j] = 1 iff the agent can take action j from
            state i.
        """

        q_values = self.q_network_target(states)

        # Set q-values of impossible actions to a very large negative number.
        inverse_pna = 1 - possible_actions
        impossible_action_penalty = self.ACTION_NOT_POSSIBLE_VAL * inverse_pna
        q_values += impossible_action_penalty
        return Variable(torch.max(q_values, 1)[0])

    def train(
        self, training_samples: TrainingDataPage, evaluator=None, episode_values=None
    ) -> None:
        self.minibatch += 1
        states = Variable(torch.from_numpy(training_samples.states).type(self.dtype))
        actions = Variable(
            torch.from_numpy(training_samples.actions).type(self.dtypelong)
        )
        rewards = Variable(torch.from_numpy(training_samples.rewards).type(self.dtype))
        next_states = Variable(
            torch.from_numpy(training_samples.next_states).type(self.dtype)
        )
        possible_next_actions = Variable(
            torch.from_numpy(training_samples.possible_next_actions).type(self.dtype)
        )
        time_diffs = torch.tensor(training_samples.time_diffs).type(self.dtype)
        discount_tensor = torch.tensor(
            np.array([self.gamma for x in range(len(rewards))])
        ).type(self.dtype)
        not_done_mask = Variable(
            torch.from_numpy(training_samples.not_terminals.astype(int))
        ).type(self.dtype)

        if self.use_seq_num_diff_as_time_diff:
            discount_tensor = discount_tensor.pow(time_diffs)

        # Compute max a' Q(s', a') over all possible actions using target network
        max_q_values = self.get_max_q_values(next_states, possible_next_actions)
        filtered_max_q_vals = max_q_values * not_done_mask

        if self.minibatch >= self.reward_burnin:
            target_q_values = rewards + (discount_tensor * filtered_max_q_vals)
        else:
            target_q_values = rewards

        # Get Q-value of action taken
        q_values = (
            self.q_network(states).gather(1, actions.argmax(1).unsqueeze(1)).squeeze()
        )

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

        model_propensities, model_values_on_logged_actions = None, None
        if all_action_scores is not None:
            model_propensities = Evaluator.softmax(
                all_action_scores, self.rl_temperature
            )
            if logged_actions is not None:
                model_values_on_logged_actions = np.sum(
                    (logged_actions * all_action_scores), axis=1, keepdims=True
                )

        evaluator.report(
            td_loss,
            logged_actions,
            logged_propensities,
            logged_rewards,
            logged_values,
            model_propensities,
            all_action_scores,
            model_values_on_logged_actions,
        )

    def internal_prediction(self, states):
        """ Returns list of Q-values output from Q-network
        :param states states as list of states to produce actions for
        """
        self.q_network.eval()
        with torch.no_grad():
            states = Variable(torch.from_numpy(np.array(states)).type(self.dtype))
            q_values = self.q_network(states)
        self.q_network.train()
        return q_values.cpu().data.numpy()
