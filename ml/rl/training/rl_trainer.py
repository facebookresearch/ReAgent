#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import workspace
import numpy as np

import logging
logger = logging.getLogger(__name__)

from ml.rl.preprocessing.normalization import preprocess_feature
from ml.rl.training.ml_trainer import MLTrainer
from ml.rl.training.target_network import TargetNetwork


def normalize_features(input_vectors, features, _input_normalization_parameter):
    if _input_normalization_parameter is not None:
        for idx, feature in enumerate(features):
            input_vectors[:, idx] = preprocess_feature(
                input_vectors[:, idx], _input_normalization_parameter[feature]
            )
    return input_vectors


def replace_nans(inputs):
    if np.any(np.isnan(inputs)):
        logger.info("normalization NAN: " + str(np.where(np.isnan(inputs))))
        inputs[np.where(np.isnan(inputs))] = 0.
    return inputs


class RLTrainer(MLTrainer):
    def __init__(self, state_normalization_parameters, parameters):
        print(state_normalization_parameters)
        print(parameters)

        self._state_features = list(state_normalization_parameters.keys())
        self._state_normalization_parameters = state_normalization_parameters

        MLTrainer.__init__(self, "rl_trainer", parameters.training)
        self.target_network = TargetNetwork(
            self, parameters.rl.target_update_rate
        )

        self.reward_burnin = parameters.rl.reward_burnin
        self.maxq_learning = parameters.rl.maxq_learning
        self.rl_discount_rate = parameters.rl.gamma

        self._iteration = 0
        self._buffers = None
        self.minibatch_size = parameters.training.minibatch_size

    def _prepare_states(self, states):
        """
        Normalizes input states and replaces NaNs with 0. Returns a matrix of
        the same shape.

        :param states: Numpy array with shape (batch_size, state_dim) containing
            raw state inputs
        """
        normalized_states = normalize_features(
            states, self._state_features, self._state_normalization_parameters
        )
        return replace_nans(normalized_states)

    def get_state_features(self):
        return self._state_features

    @property
    def num_state_features(self):
        return len(self._state_features)

    def predictor(self):
        """
        Builds a DiscreteActionTrainerMultipleOutputPredictor using the networks
        trained by this Trainer.
        """
        raise Exception("Virtual")

    def stream_df(self, df, evaluator=None):
        """Load large batch as training set. This batch will further be broken
        down into minibatches

        :param df a batch of data.
        """
        raise Exception("Virtual")

    def stream(
        self, current_states, actions, rewards, next_states, next_actions,
        terminals, possible_next_actions, reward_timelines, evaluator
    ):
        """Load large batch as training set. This batch will further be broken
        down into minibatches
        """

        if rewards.ndim == 1:
            rewards = rewards.reshape(-1, 1)
        if terminals.ndim == 1:
            terminals = terminals.reshape(-1, 1)

        num_buffers = 8
        if self._buffers is not None and self._buffers[0].shape[0] > 0:
            actions = np.concatenate([self._buffers[0], actions])
            current_states = np.concatenate([self._buffers[1], current_states])
            rewards = np.concatenate([self._buffers[2], rewards])
            next_states = np.concatenate([self._buffers[3], next_states])
            if next_actions is not None:
                next_actions = np.concatenate([self._buffers[4], next_actions])
            terminals = np.concatenate([self._buffers[5], terminals])
            if possible_next_actions is not None:
                possible_next_actions = np.concatenate(
                    [self._buffers[6], possible_next_actions]
                )
            if reward_timelines is not None:
                reward_timelines = np.concatenate(
                    [self._buffers[7], reward_timelines]
                )

        self._buffers = None

        page_size = current_states.shape[0]

        for batch_start in range(0, page_size, self.minibatch_size):
            batch_end = batch_start + self.minibatch_size
            if page_size < batch_end:
                break
                self._buffers = [[] for _ in range(num_buffers)]
                self._buffers[0] = actions[batch_start:batch_end]
                self._buffers[1] = current_states[batch_start:batch_end]
                self._buffers[2] = rewards[batch_start:batch_end]
                self._buffers[3] = next_states[batch_start:batch_end]
                if next_actions:
                    self._buffers[4] = next_actions[batch_start:batch_end]
                self._buffers[5] = terminals[batch_start:batch_end]
                if possible_next_actions:
                    self._buffers[6] = possible_next_actions[batch_start:
                                                             batch_end]
                if reward_timelines:
                    self._buffers[7] = reward_timelines[batch_start:batch_end]
            else:
                na = None
                if next_actions is not None:
                    na = next_actions[batch_start:batch_end]
                pna = None
                if possible_next_actions is not None:
                    pna = possible_next_actions[batch_start:batch_end]
                rt = None
                if reward_timelines is not None:
                    rt = reward_timelines[batch_start:batch_end]
                predictions, td_loss = self.train(
                    current_states[batch_start:batch_end],
                    actions[batch_start:batch_end],
                    rewards[batch_start:batch_end],
                    next_states[batch_start:batch_end],
                    na,
                    terminals[batch_start:batch_end],
                    pna,
                )
                if evaluator is not None:
                    evaluator.report(
                        rt,
                        self.get_maxq_labels(
                            current_states[batch_start:batch_end],
                            actions[batch_start:batch_end]
                        ), td_loss
                    )
            self._iteration += 1

    def train(
        self,
        states,
        actions,
        rewards,
        next_states,
        next_actions,
        terminals,
        possible_next_actions,
    ):
        """
        Takes in a batch of transitions. For transition i, calculates target qval:
            next_q_values_i = {
                max_a Q(next_state_i, a),  self.max_qlearning
                Q(next_state_i, next_action_i), !self.max_qlearning
            }
            q_val_target_i = {
                r_i, terminals_i
                r_i + gamma * next_q_values_i, !terminals_i
            }
        Trains Q Network on the q_val_targets as labels.

        :param states: Numpy array with shape (batch_size, state_dim). The ith
            row is a representation of the ith transition's state.
        :param actions: It is a Numpy array with
            shape (batch_size, action_dim). The ith row contains the one-hotted
            representation of the ith transition's action: actions[i][j] = 1 if
            action_i == j else 0.
        :param rewards: Numpy array with shape (batch_size, 1). The ith entry is
            the reward experienced at the ith transition.
        :param terminals: Numpy array with shape (batch_size, 1). The ith entry
            is equal to 1 iff the ith transition's state is terminal.
        :param next_states: Numpy array with shape (batch_size, state_dim). The
            ith row is a representation of the ith transition's next state.
        :param next_actions: It is a Numpy array with
            shape (batch_size, action_dim). The ith row contains the one-hotted
            representation of the ith transition's action: next_actions[i][j] = 1 if
            next_action_i == j else 0.
        :param possible_next_actions: Numpy array with shape (batch_size, action_dim).
            possible_next_actions[i][j] = 1 iff the agent can take action j from
            state i.
        """

        batch_size = states.shape[0]
        assert states.shape == (batch_size, self.num_state_features)
        assert rewards.shape == (batch_size, 1)
        assert rewards.dtype == np.float32
        assert next_states.shape == (batch_size, self.num_state_features)
        assert terminals.shape == (batch_size, 1)

        q_vals_target = np.copy(rewards)
        if self._iteration >= self.reward_burnin:
            if self.maxq_learning:
                next_q_values = self.get_maxq_labels(
                    next_states, possible_next_actions
                )
            else:
                next_q_values = self.get_sarsa_labels(next_states, next_actions)

            q_vals_target += (
                1.0 - terminals
            ) * self.rl_discount_rate * next_q_values

        self.run_train_rl_nn(states, actions, q_vals_target)

        if self._iteration >= self.reward_burnin:
            self.target_network.enable_slow_updates()
        self.target_network.target_update()
        self._iteration += 1

        return (
            np.squeeze(workspace.FetchBlob(self.q_values)),
            workspace.FetchBlob(self.loss_blob)
        )
