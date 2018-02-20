#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from typing import Optional, Union

from caffe2.python import workspace

import logging
logger = logging.getLogger(__name__)

from ml.rl.thrift.core.ttypes import DiscreteActionModelParameters,\
    ContinuousActionModelParameters
from ml.rl.training.evaluator import Evaluator
from ml.rl.training.ml_trainer import MLTrainer
from ml.rl.training.target_network import TargetNetwork
from ml.rl.training.training_data_page import TrainingDataPage


class RLTrainer(MLTrainer):
    def __init__(
        self,
        parameters: Union[DiscreteActionModelParameters,
                          ContinuousActionModelParameters],
    ) -> None:
        logger.info(str(parameters))

        assert parameters.training.layers[0] >= 0,\
            "Set layers[0] to a the number of features"

        self.num_features = parameters.training.layers[0]

        MLTrainer.__init__(self, "rl_trainer", parameters.training)

        self.target_network = TargetNetwork(
            self, parameters.rl.target_update_rate
        )

        self.reward_burnin = parameters.rl.reward_burnin
        self.maxq_learning = parameters.rl.maxq_learning
        self.rl_discount_rate = parameters.rl.gamma

        self.training_iteration = 0
        self.minibatch_size = parameters.training.minibatch_size

    @property
    def sarsa(self) -> bool:
        """
        Returns whether or not this trainer generates target values using SARSA.
        """
        return not self.maxq_learning

    def stream_tdp(
        self, tdp: TrainingDataPage, evaluator: Optional[Evaluator] = None
    ) -> None:
        """
        Loads a large batch of transitions from a page of training data. This
        batch will be further broken down into minibatches for training.

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
        self, states, actions, rewards, next_states, next_actions,
        not_terminals, possible_next_actions, reward_timelines, evaluator
    ):
        """
        Load large batch as training set. This batch will be broken down into
        minibatches. Assumes that states, next_states, and actions (in the
        parametric action case) need no further normalization.
        """

        assert rewards.ndim == 2
        assert not_terminals.ndim == 2

        page_size = states.shape[0]
        assert page_size == self.minibatch_size

        self.train(
            states,
            actions,
            rewards,
            next_states,
            next_actions,
            not_terminals,
            possible_next_actions,
        )
        if evaluator is not None:
            evaluator.report(
                reward_timelines,
                self.get_q_values(states, actions),
                workspace.FetchBlob(self.loss_blob),
            )

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        next_actions: Optional[np.ndarray],
        not_terminals: np.ndarray,
        possible_next_actions,
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

        batch_size = self.minibatch_size
        assert rewards.shape == (
            batch_size, 1
        ), "Invalid reward shape: " + \
            str(rewards.shape) + " != " + str(self.minibatch_size)
        assert rewards.dtype == np.float32
        assert not_terminals.shape == (
            batch_size, 1
        ), 'terminals invalid ' + str(not_terminals.shape)

        q_vals_target = np.copy(rewards)
        if self.training_iteration >= self.reward_burnin:
            if self.training_iteration == self.reward_burnin:
                logger.info(
                    "Minibatch number == reward_burnin. Starting RL updates."
                )
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
