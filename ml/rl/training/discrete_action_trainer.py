from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from caffe2.python import workspace

import logging
logger = logging.getLogger(__name__)

from ml.rl.training.discrete_action_predictor import DiscreteActionPredictor
from ml.rl.training.ml_trainer import MLTrainer
from ml.rl.training.ml_trainer import GenerateLossOps
from ml.rl.training.rl_trainer import RLTrainer


class DiscreteActionTrainer(RLTrainer):
    # Set to a very large negative number.  Guaranteed to be worse than any
    #     legitimate action
    ACTION_NOT_POSSIBLE_VAL = -1e20

    def __init__(self, state_normalization_parameters, parameters):
        self._actions = parameters.actions
        if parameters.training.layers[0] is None or\
           parameters.training.layers[0] == -1:
            parameters.training.layers[0] = len(state_normalization_parameters)

        # There is a logical 1-dimensional output for each state/action pair,
        # but the underlying network computes num_actions-dimensional outputs
        if parameters.training.layers[-1] in [None, -1, 1]:
            parameters.training.layers[-1] = self.num_actions

        assert parameters.training.layers[-1] == self.num_actions,\
            "Set layers[-1] to a the number of actions or a default placeholder value"

        RLTrainer.__init__(self, state_normalization_parameters, parameters)

    def get_actions(self):
        return self._actions

    @property
    def num_actions(self):
        return len(self._actions)

    def stream_df(self, df, evaluator):
        """Load large batch as training set. This batch will further be broken
        down into minibatches

        :param df a batch of data.
        """
        # terminal states have no next_action=1
        is_terminal = df.next_action.sum(axis=1) < 1e-6
        return self.stream(
            self._prepare_states(df.state_features), df.action, df.reward,
            self._prepare_states(df.next_state_features), df.next_action,
            is_terminal, df.possible_next_actions, df.reward_timelines,
            evaluator
        )

    def _setup_initial_blobs(self):
        self.input_dim = self.num_state_features
        self.output_dim = self.num_actions

        self.action_blob = "action"
        workspace.FeedBlob(self.action_blob, np.zeros(1, dtype=np.float32))

        MLTrainer._setup_initial_blobs(self)

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
        :param actions: Numpy array with shape (batch_size, action_dim). The ith
            row contains the one-hotted representation of the ith transition's
            action: actions[i][j] = 1 if action_i == j else 0.
        :param rewards: Numpy array with shape (batch_size, 1). The ith entry is
            the reward experienced at the ith transition.
        :param terminals: Numpy array with shape (batch_size, 1). The ith entry
            is equal to 1 iff the ith transition's state is terminal.
        :param next_states: Numpy array with shape (batch_size, state_dim). The
            ith row is a representation of the ith transition's next state.
        :param next_actions: Numpy array with shape (batch_size, action_dim). The
            ith row contains the one-hotted representation of the ith transition's
            action: next_actions[i][j] = 1 if next_action_i == j else 0.
        :param possible_next_actions: Numpy array with shape (batch_size, action_dim).
            possible_next_actions[i][j] = 1 iff the agent can take action j from
            state i.
        """

        batch_size = states.shape[0]
        assert actions.shape == (batch_size, self.num_actions)
        assert next_states.shape == (batch_size, self.num_state_features)
        assert terminals.shape == (batch_size, 1)
        if next_actions is not None:
            assert next_actions.shape == (batch_size, self.num_actions)
        if possible_next_actions is not None:
            assert possible_next_actions.shape == (batch_size, self.num_actions)
        RLTrainer.train(
            self,
            states,
            actions,
            rewards,
            next_states,
            next_actions,
            terminals,
            possible_next_actions,
        )

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
        self.q_values = self.train_model.net.ExpandDims(q_val_select, dims=[1])

        GenerateLossOps(
            self.train_model, self.model_id, self.labels_blob, self.q_values,
            self.loss_blob
        )

    def update_model(self, states, actions, q_vals_target):
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
        workspace.FeedBlob(self.input_blob, states)
        workspace.FeedBlob(self.action_blob, actions)
        workspace.FeedBlob(self.labels_blob, q_vals_target)
        workspace.RunNet(self.train_model.net)

    def get_max_q_values(
        self, next_states, possible_next_actions=None, use_target_network=True
    ):
        """
        Takes in an array of next_states and outputs an array of the same shape
        whose ith entry = max_{possible_next_actions} Q(state_i, a).

        :param next_states: Numpy array with shape (batch_size, state_dim). Each
            row contains a representation of a state.
        :param possible_next_actions: Numpy array with shape (batch_size, action_dim).
            possible_next_actions[i][j] = 1 iff the agent can take action j from
            state i.
        :use_target_network: Boolean that indicates whether or not to use this
            trainer's TargetNetwork to compute Q values.
        """
        q_values = self.get_q_values_all_actions(next_states, use_target_network)

        if possible_next_actions is not None:
            mask = np.multiply(
                np.logical_not(possible_next_actions),
                self.ACTION_NOT_POSSIBLE_VAL
            )
            q_values += mask

        return np.max(q_values, axis=1, keepdims=True)

    def get_q_values(self, states, actions):
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
        self, states, use_target_network=True
    ):
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
        else:
            workspace.FeedBlob(self.input_blob, states.astype(np.float32))
            workspace.RunNet(self.score_model.net)
            return workspace.FetchBlob(self.output_blob)

    def get_sarsa_values(self, next_states, next_actions):
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

    def predictor(self):
        """
        Builds a DiscreteActionPredictor using the MLTrainer underlying this
        DiscreteActionTrainer.
        """
        return DiscreteActionPredictor.from_trainers(
            self, self._state_features, self._actions,
            self._state_normalization_parameters
        )

    def get_policy(self, state):
        """
        Returns the action with the highest approximated q-value for the given
        state.
        """
        q_values = self.get_q_values_all_actions(np.array([state]), False)
        return np.argmax(q_values[0])
