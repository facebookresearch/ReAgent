'''
Copyright (c) 2017-present, Facebook, Inc.
All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
'''

# @package dqn
# Module rl2_caffe2.rlmodels.dqn
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from caffe2.python import workspace

from ml.rl.test.gym.rlmodels.rlmodel_base_adapted import RLNN_Adapted
from ml.rl.test.gym.rlmodels.rlmodel_helper import \
    DEFAULT_EPSILON, epsilon_greedy_onehot
from ml.rl.training.ml_trainer import GenerateLossOps


#  DQN-max-action: Deep Q Network
#   [DQN-Atari](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) (deepmind)
#  * input: state
#  * output: Qval_a(vector of A-dimension)
class DQN_rlnn_Adapted(RLNN_Adapted):
    def init_params(self):
        self.epsilon = DEFAULT_EPSILON

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
        q_val_select_reshaped = self.train_model.net.ExpandDims(
            q_val_select, dims=[1]
        )

        GenerateLossOps(
            self.train_model, self.model_id, self.labels_blob,
            q_val_select_reshaped, self.loss_blob
        )

    def run_score_rl_nn(self, states):
        """
        Takes in a set of states and runs the test Q Network on them.

        Creates Q(states, actions), a blob with shape (batch_size, action_dim).
        Q(states, actions)[i][j] is an approximation of Q*(states[i], action_j).
        Note that action_j takes on every possible action (of which there are
        self.action_dim_. Stores blob in self.output_blob.

        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        """
        workspace.FeedBlob(self.input_blob, states.astype(np.float32))
        workspace.RunNet(self.score_model.net)

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
        workspace.FeedBlob(self.input_blob, states)
        workspace.FeedBlob(self.action_blob, actions)
        workspace.FeedBlob(self.labels_blob, q_vals_target.astype(np.float32))
        workspace.RunNet(self.train_model.net)

    def get_q_val_best(self, states):
        """
        Takes in an array of states and outputs an array of the same shape whose
        ith entry is the action a that maximizes Q(state_i, a).

        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        """
        _, q_values_max = \
            self._get_action_policy_batch(states)
        return q_values_max

    def get_q_val_action(self, states, actions):
        """
        Takes in a set of states and corresponding actions. For each
        (state_i, action_i) pair, calculates Q(state, action). Returns these q
        values in a Numpy array of shape (batch_size, 1).

        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        :param actions: Numpy array with shape (batch_size, action_dim)
        """
        self.run_score_rl_nn(states)
        q_values = workspace.FetchBlob(self.output_blob)
        q_val_select = q_values[actions == 1]
        q_val_select_reshaped = np.expand_dims(q_val_select, 1)
        return q_val_select_reshaped

    def _get_action_policy_batch(self, states):
        """
        Takes in a set of states and, for each, outputs the action a that maximizes
        Q(state_i, a) and the value Q(state_i, a) itself. Returns the results in
        two numpy arrays, both of shape (batch_size, 1).

        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        """
        self.run_score_rl_nn(states)
        q_values = workspace.FetchBlob(self.output_blob)
        q_values_max = np.max(q_values, axis=1, keepdims=True)
        act_all = (q_values - q_values_max == 0).astype(np.int32)
        return act_all, q_values_max

    def get_policy(self, state, policy_test=False):
        """
        Takes in an array of states and outputs an array of the same shape whose
        ith entry is max_a Q(state_i, a).

        :param states: Numpy array with shape (batch_size, state_dim). Each row
            contains a representation of a state.
        :param no_explore: Boolean that controls whether or not to bypass an
            epsilon-greedy action selection policy.
        """
        states = np.array([state], dtype=np.float32)
        act_all, _ = self._get_action_policy_batch(states)
        action = act_all[0]
        if not policy_test:
            action = epsilon_greedy_onehot(
                action, self.output_dim, self.epsilon
            )
        return action
