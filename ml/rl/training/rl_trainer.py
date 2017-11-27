#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from typing import List, Dict, Optional

from caffe2.python import workspace, core

import logging
logger = logging.getLogger(__name__)

from ml.rl.preprocessing.normalization import NormalizationParameters
from ml.rl.preprocessing.preprocessor_net import prepare_normalization,\
    normalize_dense_matrix
from ml.rl.training.evaluator import Evaluator
from ml.rl.training.ml_trainer import MLTrainer
from ml.rl.training.rl_predictor import RLPredictor
from ml.rl.training.target_network import TargetNetwork
from ml.rl.training.training_data_page import TrainingDataPage


class RLTrainer(MLTrainer):
    def __init__(
        self,
        state_normalization_parameters: Dict[str, NormalizationParameters],
        parameters
    ) -> None:
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

        self.training_iteration = 0
        self._buffers = None
        self.minibatch_size = parameters.training.minibatch_size

        self._prepare_state_normalization()

    def _normalize_states(self, states: np.ndarray) -> np.ndarray:
        """
        Normalizes input states and replaces NaNs with 0. Returns a matrix of
        the same shape. Make sure to have set up the underlying normalization net
        with `_prepare_state_normalization`.

        :param states: Numpy array with shape (batch_size, state_dim) containing
            raw state inputs
        """
        return normalize_dense_matrix(
            states, self.num_state_features, self.state_norm_blobs,
            self.state_norm_net, self.state_norm_blobname_template
        )

    def _prepare_state_normalization(self):
        """
        Sets up operators for action normalization net.
        """
        self.state_norm_net = core.Net("state_norm_net")
        self.state_norm_blobname_template = '{}_input_state'
        self.state_norm_blobs = prepare_normalization(
            self.state_norm_net, self._state_normalization_parameters,
            self._state_features, self.state_norm_blobname_template
        )

    def get_state_features(self) -> List[str]:
        return self._state_features

    @property
    def num_state_features(self) -> int:
        return len(self._state_features)

    @property
    def sarsa(self) -> bool:
        """
        Returns whether or not this trainer generates target values using SARSA.
        """
        return not self.maxq_learning

    def predictor(self) -> RLPredictor:
        """
        Builds a Predictor using the networks undrlying this Trainer.
        """
        raise NotImplementedError()

    def stream_df(
        self, tdp: TrainingDataPage, evaluator: Optional[Evaluator] = None
    ) -> None:
        """
        Loads a large batch of transitions from a page of training data. This
        batch will further be broken down into minibatches for training.

        :param tdp: TrainingDataPage object that supplies transitions.
        :param evaluator: Evaluator object to record TD and compute MC losses.
        """
        raise NotImplementedError()

    def get_max_q_values(
        self, next_states: np.ndarray, possible_next_actions
    ) -> np.ndarray:
        """
        Takes in an array of next_states and outputs an array of the same shape
        whose ith entry = max_{pna} Q(state_i, pna). Uses target network for
        Q(state_i, pna) approximation.

        :param next_states: Numpy array with shape (batch_size, state_dim). Each
            row contains a representation of a state.
        :param possible_next_actions: See subclass' `get_max_q_values` documentation.
        """
        raise NotImplementedError()

    def get_sarsa_values(
        self, next_states: np.ndarray, next_actions: np.ndarray
    ) -> np.ndarray:
        """
        Takes in a set of next_states and corresponding next_actions. For each
        (next_state_i, next_action_i) pair, calculates Q(next_state, next_action).
        Returns these q values in a Numpy array of shape (batch_size, 1).

        :param next_states: Numpy array with shape (batch_size, state_dim). The
            ith row is a representation of the ith transition's next_state.
        :param next_actions: See subclass' `get_sarsa_values` documentation.
        """
        raise NotImplementedError()

    def update_model(
        self, states: np.ndarray, actions: np.ndarray, q_vals_target: np.ndarray
    ) -> None:
        """
        Takes in states, actions, and target q values. Updates the model:
            Runs the forward pass, computing Q(states, actions).
                Q(states, actions)[i][j] is an approximation of Q*(states[i], action_j).
            Comptutes Loss of Q(states, actions) with respect to q_vals_targets.
            Updates Q Network's weights according to loss and optimizer.

        :param states: Numpy array with shape (batch_size, state_dim). The ith
            row is a representation of the ith transition's state.
        :param actions: Numpy array with shape (batch_size, action_dim). The ith
            row is a representation of the ith transition's action.
        :param q_vals_targets: Numpy array with shape (batch_size, 1). The ith
            row is the label to train against for the data from the ith transition.
        """
        raise NotImplementedError()

    def stream(
        self, current_states, actions, rewards, next_states, next_actions,
        not_terminals, possible_next_actions, reward_timelines, evaluator
    ):
        """Load large batch as training set. This batch will further be broken
        down into minibatches
        """

        if rewards.ndim == 1:
            rewards = rewards.reshape(-1, 1)
        if not_terminals.ndim == 1:
            not_terminals = not_terminals.reshape(-1, 1)

        use_next_actions = next_actions is not None and self.sarsa
        use_pna = possible_next_actions is not None and self.maxq_learning
        use_rt = reward_timelines is not None

        num_buffers = 8
        if self._buffers is not None and self._buffers[0].shape[0] > 0:
            actions = np.concatenate([self._buffers[0], actions])
            current_states = np.concatenate([self._buffers[1], current_states])
            rewards = np.concatenate([self._buffers[2], rewards])
            next_states = np.concatenate([self._buffers[3], next_states])
            if use_next_actions:
                next_actions = np.concatenate([self._buffers[4], next_actions])
            not_terminals = np.concatenate([self._buffers[5], not_terminals])
            if use_pna:
                possible_next_actions = np.concatenate(
                    [self._buffers[6], possible_next_actions]
                )
            if use_rt:
                reward_timelines = np.concatenate(
                    [self._buffers[7], reward_timelines]
                )

        self._buffers = None
        page_size = current_states.shape[0]

        for batch_start in range(0, page_size, self.minibatch_size):
            batch_end = batch_start + self.minibatch_size
            if page_size < batch_end:
                self._buffers = [[] for _ in range(num_buffers)]
                self._buffers[0] = actions[batch_start:]
                self._buffers[1] = current_states[batch_start:]
                self._buffers[2] = rewards[batch_start:]
                self._buffers[3] = next_states[batch_start:]
                if use_next_actions:
                    self._buffers[4] = next_actions[batch_start:]
                self._buffers[5] = not_terminals[batch_start:]
                if use_pna:
                    self._buffers[6] = possible_next_actions[batch_start:]
                if use_rt:
                    self._buffers[7] = reward_timelines[batch_start:]
            else:
                na_batch = (
                    next_actions[batch_start:batch_end] if use_next_actions
                    else None
                )
                pna_batch = (
                    possible_next_actions[batch_start:batch_end] if use_pna
                    else None
                )
                rt_batch = (
                    reward_timelines[batch_start:batch_end] if use_rt else None
                )
                current_states_batch = current_states[batch_start:batch_end]
                actions_batch = actions[batch_start:batch_end]
                self.train(
                    current_states_batch,
                    actions_batch,
                    rewards[batch_start:batch_end],
                    next_states[batch_start:batch_end],
                    na_batch,
                    not_terminals[batch_start:batch_end],
                    pna_batch
                )
                if evaluator is not None:
                    evaluator.report(
                        rt_batch, self.get_q_values(
                            current_states_batch, actions_batch
                        ), workspace.FetchBlob(self.loss_blob)
                    )

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        next_actions: Optional[np.ndarray],
        not_terminals: np.ndarray,
        possible_next_actions: Optional[List],
    ) -> None:
        """
        Takes in a batch of transitions. For transition i, calculates target qval:
            next_q_values_i = {
                max_{pna_i} Q(next_state_i, pna_i), self.maxq_learning
                Q(next_state_i, next_action_i), self.sarsa
            }
            q_val_target_i = {
                r_i + gamma * next_q_values_i, not_terminals_i
                r_i, !not_terminals_i
            }
        Trains Q Network on the q_val_targets as labels.

        :param states: Numpy array with shape (batch_size, state_dim). The ith
            row is a representation of the ith transition's state.
        :param actions: See subclass' `train` documentation.
        :param rewards: Numpy array with shape (batch_size, 1). The ith entry is
            the reward experienced at the ith transition.
        :param not_terminals: Numpy array with shape (batch_size, 1). The ith entry
            is equal to 1 iff the ith transition's state is not terminal.
        :param next_states: Numpy array with shape (batch_size, state_dim). The
            ith row is a representation of the ith transition's next state.
        :param next_actions: See subclass' `train` documentation.
        :param possible_next_actions: See subclass' `train` documentation.
        """

        batch_size = states.shape[0]
        assert states.shape == (batch_size, self.num_state_features)
        assert rewards.shape == (batch_size, 1)
        assert rewards.dtype == np.float32
        assert next_states.shape == (batch_size, self.num_state_features)
        assert not_terminals.shape == (batch_size, 1)

        q_vals_target = np.copy(rewards)
        if self.training_iteration >= self.reward_burnin:
            if self.training_iteration == self.reward_burnin:
                logger.info("Minibatch number == reward_burnin. Starting RL updates.")
            if self.maxq_learning:
                next_q_values = self.get_max_q_values(
                    next_states, possible_next_actions
                )
            else:
                next_q_values = self.get_sarsa_values(next_states, next_actions)

            q_vals_target += not_terminals * self.rl_discount_rate * next_q_values

        self.update_model(states, actions, q_vals_target)

        if self.training_iteration >= self.reward_burnin:
            self.target_network.enable_slow_updates()
        self.target_network.target_update()
        self.training_iteration += 1
