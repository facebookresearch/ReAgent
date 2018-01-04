#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Dict, Optional
import numpy as np

from caffe2.python import workspace

import logging
logger = logging.getLogger(__name__)

from ml.rl.preprocessing.normalization import NormalizationParameters,\
    get_num_output_features
from ml.rl.thrift.core.ttypes import DiscreteActionModelParameters
from ml.rl.training.discrete_action_predictor import DiscreteActionPredictor
from ml.rl.training.evaluator import Evaluator
from ml.rl.training.ml_trainer import GenerateLossOps
from ml.rl.training.ml_trainer import MLTrainer
from ml.rl.training.rl_trainer import RLTrainer
from ml.rl.training.training_data_page import TrainingDataPage


class DiscreteActionTrainer(RLTrainer):
    # Set to a very large negative number.  Guaranteed to be worse than any
    #     legitimate action
    ACTION_NOT_POSSIBLE_VAL = -1e20

    def __init__(
        self,
        state_normalization_parameters: Dict[str, NormalizationParameters],
        parameters: DiscreteActionModelParameters,
        skip_normalization: Optional[bool] = False
    ) -> None:
        self._actions = parameters.actions
        self.num_processed_state_features = get_num_output_features(
            state_normalization_parameters
        )
        if parameters.training.layers[0] in [None, -1, 1]:
            parameters.training.layers[0] = self.num_state_features

        # There is a logical 1-dimensional output for each state/action pair,
        # but the underlying network computes num_actions-dimensional outputs
        if parameters.training.layers[-1] in [None, -1, 1]:
            parameters.training.layers[-1] = self.num_actions

        assert parameters.training.layers[-1] == self.num_actions,\
            "Set layers[-1] to a the number of actions or a default placeholder value"

        RLTrainer.__init__(
            self, state_normalization_parameters, parameters, skip_normalization
        )

    @property
    def num_state_features(self) -> int:
        return self.num_processed_state_features

    @property
    def num_actions(self) -> int:
        return 0 if self._actions is None else len(self._actions)

    def stream_tdp(
        self, tdp: TrainingDataPage, evaluator: Optional[Evaluator] = None
    ) -> None:
        """
        Loads a large batch of transitions from a page of training data. This
        batch will further be broken down into minibatches for training.

        :param tdp: TrainingDataPage object that supplies transitions.
        :param evaluator: Evaluator object to record TD and compute MC losses.
        """
        not_terminals = tdp.not_terminals
        if not_terminals is None:
            # Terminal states' corresponding next_action vectors' values are all 0
            not_terminals = tdp.next_actions.sum(axis=1) >= 1e-6

        # If we encounter GPU out of memory errors, normalize within minibatches
        self.stream(
            self._normalize_states(tdp.states), tdp.actions, tdp.rewards,
            self._normalize_states(tdp.next_states), tdp.next_actions,
            not_terminals, tdp.possible_next_actions, tdp.reward_timelines,
            evaluator
        )

    def _setup_initial_blobs(self):
        self.input_dim = self.num_state_features
        self.output_dim = self.num_actions

        self.action_blob = "action"
        workspace.FeedBlob(self.action_blob, np.zeros(1, dtype=np.float32))

        MLTrainer._setup_initial_blobs(self)

    def _validate_train_inputs(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        next_actions: Optional[np.ndarray],
        not_terminals: np.ndarray,
        possible_next_actions: np.ndarray,
    ):
        batch_size = self.minibatch_size
        assert states.shape == (batch_size, self.num_state_features)
        assert actions.shape == (batch_size, self.num_actions)
        assert next_states.shape == (batch_size, self.num_state_features)
        assert not_terminals.shape == (batch_size, 1)
        if next_actions is not None:
            assert next_actions.shape == (batch_size, self.num_actions)
        if possible_next_actions is not None:
            assert possible_next_actions.shape == (batch_size, self.num_actions)

    def _generate_train_model_loss(self):
        """
        Computes Q(state_i, action_i) for each transition i in the batch
        being trained on.
            Does an elementwise multiplication of Q(states, actions), stored in
            self.output_blob, and the one-hotted action matrix stored in
            self.action_blob. This results in a matrix M such that
            M_{i,j} = 0 if action_i != j else Q(state_i, action_j).

            Sums across each row of M. Each should only have one non-zero entry.
            This results in an array q_val_select with shape (1, batch_size) such
            that q_val_select[i] = Q(state_i, action_i).

            Sets q_val_select_shaped to the transpose of q_val_select. This now
            contains Q(state_i, action_i) for each transition i in the batch.

        Computes loss of q_vals_targets, stored in self.labels_blob, with
        respect to Q(state_i, action_i), and stores the result in self.loss_blob.
        """
        q_val_select = self.train_model.net.Mul(
            [self.output_blob, self.action_blob],
        ).ReduceBackSum()
        q_values = self.train_model.net.ExpandDims(q_val_select, dims=[1])

        GenerateLossOps(
            self.train_model, self.model_id, self.labels_blob, q_values,
            self.loss_blob
        )

    def update_model(
        self, states: np.ndarray, actions: np.ndarray, q_vals_target: np.ndarray
    ) -> None:
        """
        Takes in states, actions, and target q values. Updates the model:

            Runs the forward pass, computing Q(states, actions).
                Q(states, actions)[i][j] is an approximation of Q*(states[i], action_j).
            Comptutes Loss of Q(states, actions) with respect to q_vals_targets
            Updates Q Network's weights according to loss and optimizer

        :param states: Numpy array with shape (batch_size, state_dim). The ith
            row is a representation of the ith transition's state.
        :param actions: Numpy array with shape (batch_size, action_dim). The ith
            row contains the one-hotted representation of the ith action.
        :param q_vals_targets: Numpy array with shape (batch_size, 1). The ith
            row is the label to train against for the data from the ith transition.
        """
        workspace.FeedBlob(self.action_blob, actions)
        self.train_batch(states, q_vals_target)

    def get_max_q_values(
        self,
        next_states: np.ndarray,
        possible_next_actions: Optional[np.ndarray] = None,
        use_target_network: Optional[bool] = True
    ) -> np.ndarray:
        """
        Takes in an array of next_states and outputs an array of the same shape
        whose ith entry = max_{pna} Q(state_i, pna).

        :param next_states: Numpy array with shape (batch_size, state_dim). Each
            row contains a representation of a state.
        :param possible_next_actions: Numpy array with shape (batch_size, action_dim).
            possible_next_actions[i][j] = 1 iff the agent can take action j from
            state i.
        :use_target_network: Boolean that indicates whether or not to use this
            trainer's TargetNetwork to compute Q values.
        """
        q_values = self.get_q_values_all_actions(
            next_states, use_target_network
        )

        if possible_next_actions is not None:
            mask = np.multiply(
                np.logical_not(possible_next_actions),
                self.ACTION_NOT_POSSIBLE_VAL
            )
            q_values += mask

        return np.max(q_values, axis=1, keepdims=True)

    def get_q_values(
        self, states: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        """
        Takes in a set of states and actions and returns Q(states, actions).

        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        :param actions: Numpy array with shape (batch_size, action_dim). The ith
            row contains the one-hotted representation of the ith action.
        """
        q_values = self.get_q_values_all_actions(states, False)
        q_values_selected = q_values * actions
        return np.max(q_values_selected, axis=1)

    def get_q_values_all_actions(
        self, states: np.ndarray, use_target_network: Optional[bool] = True
    ) -> np.ndarray:
        """
        Takes in a set of states and runs the test Q Network on them.

        Creates Q(states, actions), a blob with shape (batch_size, action_dim).
        Q(states, actions)[i][j] is an approximation of Q*(states[i], action_j).
        Note that action_j takes on every possible action (of which there are
        self.action_dim_. Stores blob in self.output_blob and returns its value.

        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        :param possible_next_actions: Numpy array with shape (batch_size, action_dim).
            possible_next_actions[i][j] = 1 iff the agent can take action j from
            state i.
        :use_target_network: Boolean that indicates whether or not to use this
            trainer's TargetNetwork to compute Q values.
        """
        if use_target_network:
            return self.target_network.target_values(states)
        return self.score(states)

    def get_sarsa_values(
        self, next_states: np.ndarray, next_actions: np.ndarray
    ) -> np.ndarray:
        """
        Takes in a set of next_states and corresponding next_actions. For each
        (next_state_i, next_action_i) pair, calculates Q(next_state, next_action).
        Returns these q values in a Numpy array of shape (batch_size, 1).

        :param next_states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        :param actions: Numpy array with shape (batch_size, action_dim). The ith
            row contains the one-hotted representation of the ith next_action.
        """
        return self.get_max_q_values(next_states, next_actions)

    def predictor(self) -> DiscreteActionPredictor:
        """
        Builds a DiscreteActionPredictor using the MLTrainer underlying this
        DiscreteActionTrainer.
        """
        return DiscreteActionPredictor.from_trainers(
            self, self._state_features, self._actions,
            self._state_normalization_parameters
        )

    def get_policy(self, state: np.ndarray) -> int:
        """
        Returns the index of the action with the highest approximated q-value
        for the given state.

        :param state: A Numpy array of shape (state_dim, ) containing a single
            state vector. Not yet normalized.
        """
        inputs = self._normalize_states(np.array([state], dtype=np.float32))
        q_values = self.get_q_values_all_actions(inputs, False)
        return np.argmax(q_values[0])
