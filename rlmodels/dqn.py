'''
Copyright (c) 2017-present, Facebook, Inc.
All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
'''

# @package dqn
# Module rl2_caffe2.rlmodels.dqn
import numpy as np
from caffe2.python import workspace

from .rlmodel_base import RLNN
from .rlmodel_helper import add_parameter_update_ops,\
    DEFAULT_EPSILON, epsilon_greedy_onehot


#  DQN-max-action: Deep Q Network
#   [DQN-Atari](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) (deepmind)
#  * input: state
#  * output: Qval_a(vector of A-dimension)
class DQN_rlnn(RLNN):
    def init_params(self):
        self.epsilon = DEFAULT_EPSILON
        self.output_dim = self.action_dim
        # output Q(s,a) for each action

    # network construction at network level
    def construct_rl_nn(self, q_model, trainNetwork=False):
        overall_loss = None
        critic_q_values, critic_model = self.q_network(q_model, self.critic_prefix,
                                                       self.X_state, self.state_dim,
                                                       self.output_dim)
        # only critic, no action input
        overall_outputs = [critic_q_values]
        if trainNetwork:
            # loss
            critic_loss = self.create_dqn_model_loss_ops(
                critic_model, critic_q_values)
            overall_loss = [critic_loss]
            # overall gradient
            q_model.AddGradientOperators(overall_loss)
            # final generate gradient update rule
            add_parameter_update_ops(
                q_model, self.optimizer, self.learning_rate)

        return q_model, overall_outputs, overall_loss

    def create_dqn_model_loss_ops(self, model, model_outputs):
        prefix = model.Proto().name
        # mask output with one_hot action => shape = (batch_size, action_dim)
        # reduceBackSum() return sum per col => shape = (batch_size, )
        q_val_select = model.net.Mul(
            [model_outputs, self.X_action]).ReduceBackSum()
        q_val_select_2d = model.net.ExpandDims(q_val_select, dims=[1])
        dist = model.SquaredL2Distance(
            [q_val_select_2d, self.Y_qval], prefix + "_dist")
        loss = dist.AveragedLoss([], [prefix + "_loss"])
        return loss

    # network utility fucnctions for eval and train
    def eval_rl_nn(self, states):
        workspace.FeedBlob(self.X_state, states.astype(np.float32))
        workspace.RunNet(self.q_model_test.Proto().name)
        q_values_original = workspace.FetchBlob(self.critic_q_output)
        return q_values_original

    def train_rl_nn(self, states, actions, q_vals_target):
        workspace.FeedBlob(self.X_state, states)
        workspace.FeedBlob(self.X_action, actions)
        workspace.FeedBlob(self.Y_qval, q_vals_target.astype(np.float32))
        workspace.RunNet(self.q_model_train.Proto().name)
        loss = workspace.FetchBlob(self.q_model_train_loss[0])
        return loss

    # interface functions to train and predict for Trainer
    def get_q_val_action(self, states, actions):
        q_values_original = self.eval_rl_nn(states)
        q_values_actions = q_values_original[actions == 1]
        q_values_actions = np.expand_dims(q_values_actions, 1)
        return q_values_actions

    def get_q_val_best(self, states, all_possible_actions=None):
        _, q_values_max, _ = \
            self.get_action_policy_batch(states, all_possible_actions)
        return q_values_max

    def get_action_policy_batch(self, states, all_possible_actions=None):
        q_values_original = self.eval_rl_nn(states)
        q_values_final = q_values_original
        if all_possible_actions is not None:
            q_values_filter = np.where(all_possible_actions == 0, 0, -np.inf)
            q_values_final = q_values_original + q_values_filter

        q_values_max = np.max(q_values_final, axis=1, keepdims=True)
        act_max = np.argmax(q_values_final, axis=1)
        act_all = np.eye(self.action_dim)[act_max]
        return act_all, q_values_max, q_values_original

    def get_action_policy_single(self, state, policy_test=False):
        states = np.array([state], dtype=np.float32)
        act_all, q_values_max, q_values_original = self.get_action_policy_batch(
            states)
        action = act_all[0]
        qval_a = q_values_max[0]
        qval_all = q_values_original[0]
        if not policy_test:
            action = epsilon_greedy_onehot(
                action, self.action_dim, self.epsilon)
        return action, qval_a, qval_all
