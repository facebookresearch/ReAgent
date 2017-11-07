from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from caffe2.python import workspace

import logging
logger = logging.getLogger(__name__)

from ml.rl.training.continuous_action_dqn_predictor import \
    ContinuousActionDQNPredictor
from ml.rl.training.ml_trainer import MLTrainer
from ml.rl.training.rl_trainer import RLTrainer, normalize_features, \
    replace_nans


class ContinuousActionDQNTrainer(RLTrainer):
    def __init__(
        self, state_normalization_parameters, action_normalization_parameters,
        rl_parameters
    ):
        num_state_features = len(list(state_normalization_parameters.keys()))
        num_action_features = len(list(action_normalization_parameters.keys()))
        if rl_parameters.training.layers[0] is None or\
           rl_parameters.training.layers[0] == -1:
            rl_parameters.training.layers[0] = num_state_features +\
                num_action_features

        assert rl_parameters.training.layers[-1] == 1,\
            "Set layers[-1] to 1"

        self._action_features = list(action_normalization_parameters.keys())
        self._action_normalization_parameters = action_normalization_parameters
        RLTrainer.__init__(self, state_normalization_parameters, rl_parameters)
        print(action_normalization_parameters)

    def get_action_features(self):
        return self._action_features

    @property
    def num_action_features(self):
        return len(self._action_features)

    def stream_df(self, df, evaluator=None):
        """Load large batch as training set. This batch will further be broken
        down into minibatches

        :param df a batch of data.
        """
        is_terminal = np.array(
            [pna.shape[0] == 0 for pna in df.possible_next_actions],
            dtype=np.bool
        )
        return self.stream(
            self._prepare_states(df.state_features),
            self._prepare_actions(df.action), df.reward,
            self._prepare_states(df.next_state_features),
            self._prepare_actions(df.next_action), is_terminal,
            df.possible_next_actions, df.reward_timelines, evaluator
        )

    def _prepare_actions(self, actions):
        """
        Normalizes input states and replaces NaNs with 0. Returns a matrix of
        the same shape.

        :param states: Numpy array with shape (batch_size, state_dim) containing
            raw state inputs
        """
        normalized_actions = normalize_features(
            actions, self._action_features,
            self._action_normalization_parameters
        )
        return replace_nans(normalized_actions)

    def _setup_initial_blobs(self):
        self.input_dim = self.num_state_features + self.num_action_features
        self.output_dim = 1

        MLTrainer._setup_initial_blobs(self)

    def train(
        self, states, actions, rewards, next_states, next_actions, terminals,
        possible_next_actions
    ):
        batch_size = states.shape[0]
        assert actions.shape == (batch_size, self.num_action_features)
        if next_actions is not None:
            assert next_actions.shape == (batch_size, self.num_action_features)
        if possible_next_actions is not None:
            assert len(possible_next_actions) == batch_size
            for pna in possible_next_actions:
                if pna.shape[0] > 0:
                    assert pna.shape[1] == self.num_action_features
        return RLTrainer.train(
            self, states, actions, rewards, next_states, next_actions,
            terminals, possible_next_actions
        )

    def run_train_rl_nn(self, states, actions, q_vals_target):
        """
        Takes in states, actions, and computed target q values from a batch
        of transitions.

        Runs the training Q Network:
            Runs the forward pass, computing Q(states, actions).
                Q(states, actions)[i][j] is an approximation of Q*(states[i], action_j).
            Comptutes Loss of Q(states, actions) with respect to q_vals_targets
            Updates Q Network's weights according to loss and optimizer

        :param states: Numpy array with shape (batch_size, state_dim). The ith
            row is a representation of the ith transition's state.
        :param actions: Numpy array with shape (batch_size, action_dim). The ith
            row contains the one-hotted representation of the ith transition's
            action: actions[i][j] = 1 if action_i == j else 0
        :param q_vals_targets: Numpy array with shape (batch_size, 1). The ith
            row is the label to train against for the data from the ith transition.
        """
        workspace.FeedBlob(
            self.input_blob, np.concatenate([states, actions], axis=1)
        )
        workspace.FeedBlob(self.labels_blob, q_vals_target)
        workspace.RunNet(self.train_model.net)
        self.q_values = self.output_blob

    def get_maxq_labels(self, states, possible_next_actions):
        total_size = 0
        sizes = []
        for i in range(len(states)):
            num_possible_actions = possible_next_actions[i].shape[0]
            sizes.append(num_possible_actions)
            total_size += num_possible_actions
        inputs_to_score = np.zeros(
            [total_size, self.num_state_features + self.num_action_features],
            dtype=np.float32
        )
        cursor = 0
        num_total_features = self.num_state_features + self.num_action_features
        for i in range(len(states)):
            possible_actions = possible_next_actions[i]
            num_possible_actions = possible_actions.shape[0]
            if num_possible_actions == 0:
                continue
            cursor_end = cursor + num_possible_actions
            inputs_to_score[cursor:cursor_end, 0:self.num_state_features] \
                = np.repeat(
                    states[i].reshape(1, self.num_state_features),
                    num_possible_actions,
                    axis=0)
            inputs_to_score[cursor:cursor_end,
                            self.num_state_features:num_total_features] = \
                            possible_actions
            cursor += num_possible_actions
        all_q_values = self.target_network.target_values(inputs_to_score)
        cursor = 0
        q_values = np.zeros([len(states), 1], dtype=np.float32)
        for i in range(len(states)):
            num_possible_actions = possible_next_actions[i].shape[0]
            if num_possible_actions == 0:
                continue
            q_values[i, 0] = np.max(
                all_q_values[cursor:(cursor + num_possible_actions)]
            )
            cursor += num_possible_actions
        return q_values

    def get_sarsa_labels(self, states, actions):
        """
        Takes in a set of states and corresponding actions. For each
        (state_i, action_i) pair, calculates Q(state, action). Returns these q
        values in a Numpy array of shape (batch_size, 1).

        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        :param actions: Numpy array with shape (batch_size, action_dim).
        """
        return self.target_network.target_values(
            np.concatenate([states, actions], axis=1)
        )

    def predictor(self):
        """
        Builds a ContinuousActionPredictor using the MLTrainer underlying this
        ContinuousActionTrainer.
        """
        return ContinuousActionDQNPredictor.from_trainers(
            self, self._state_features, self._action_features,
            self._state_normalization_parameters,
            self._action_normalization_parameters
        )
