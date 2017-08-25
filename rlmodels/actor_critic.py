'''
Copyright (c) 2017-present, Facebook, Inc.
All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
'''

# @package actor-critic
# Module rl2_caffe2.rlmodels.actor_critic

import numpy as np
from caffe2.python import workspace

from .rlmodel_base import RLNN
from .rlmodel_helper import add_parameter_update_ops, deterministic_noise


#   [ActorCritic-mujoco](https://arxiv.org/pdf/1509.02971.pdf)  (deepmind)
#  * input: state + action
#  * output: action(vector of A-dimension), Qval (vector of 1-dimension)
#  * uniqueness: 2tower, critic after actor forward, for continuous action only
class ActorCritic_rlnn(RLNN):

    def init_params(self):
        self.noise_sigma = 0.3
        self.output_dim = 1
        # ActorCritic, output only one Q value for the given a Q(s,a)
        self.actor2critic_lr_scalar = 0.1
        # 0.1 # scalar for lr of actor compared to critic

    # network construction at network level
    def construct_rl_nn(self, q_model, trainNetwork=False):
        overall_loss = None
        # actor
        actor_outputs, actor_model = self.q_network(q_model, self.actor_prefix,
                                                    self.X_state,
                                                    self.state_dim,
                                                    self.action_dim)
        # critic
        critic_q_values, critic_model = self.q_network(q_model, self.critic_prefix,
                                                       self.X_state, self.state_dim,
                                                       self.output_dim,
                                                       self.X_action, self.action_dim)

        overall_outputs = [actor_outputs, critic_q_values]

        if trainNetwork:
            # losses:
            ##  critic_loss = (q_target - q)^2
            ##  actor_loss  = (-scale) * q
            weight_a_to_q = self.critic_prefix + self.action_output_label + '_w'
            weight_a_to_q_copy = weight_a_to_q + "_copy"
            actor_model.net.Copy(weight_a_to_q, weight_a_to_q_copy)

            critic_q_state_output = self.critic_prefix + self.state_output_label
            critic_q_state_output_copy = critic_q_state_output + "_copy"
            actor_model.net.Copy(critic_q_state_output,
                                 critic_q_state_output_copy)
            critic_loss = self.create_critic_model_loss_ops(critic_model,
                                                            critic_q_values,
                                                            self.Y_qval)
            actor_loss = self.create_actor_model_loss_ops(actor_model,
                                                          actor_outputs,
                                                          weight_a_to_q_copy,
                                                          critic_q_state_output_copy)
            overall_loss = [critic_loss, actor_loss]
            overall_gradient_map = q_model.AddGradientOperators(overall_loss)
            q_model.StopGradient(weight_a_to_q_copy)
            add_parameter_update_ops(
                q_model, self.optimizer, self.learning_rate)

        return q_model, overall_outputs, overall_loss

    def create_critic_model_loss_ops(self, model, critic_outputs, Y_qval):
        prefix = model.net.Proto().name
        prefix = prefix + self.critic_label
        prefix = self.critic_prefix
        dist = model.SquaredL2Distance(
            [critic_outputs, Y_qval], prefix + "_dist")
        loss = prefix + "_loss"
        loss = dist.AveragedLoss([], loss)
        return loss

    def create_actor_model_loss_ops(self, model, actor_outputs, weight_a_to_q,
                                    critic_q_state_output):
        # Actor is to maximize Q = {w_x} * x + {w_a} * a,
        # thus using (-scale) * {w_a} * a  as loss and then gradient descent
        # scale: as adjustment of learning rate for actor (0.1 suggested)
        prefix = self.actor_prefix
        q_gain_from_a = prefix + "_q_gain_from_a"
        model.MatMul([actor_outputs, model.net.Transpose(
            weight_a_to_q)], q_gain_from_a)
        q_after_gain_from_a = prefix + "_q_after_gain_from_a"
        model.net.Sum([q_gain_from_a, critic_q_state_output],
                      q_after_gain_from_a)
        neg_scaled_q_loss = prefix + "_neg_scaled_q_loss"
        model.net.Scale(q_after_gain_from_a, neg_scaled_q_loss,
                        scale=-self.actor2critic_lr_scalar)
        loss = prefix + "_loss"
        loss = model.net.AveragedLoss([neg_scaled_q_loss], loss)
        return loss

    # network utility fucnctions for eval and train
    def eval_rl_nn(self, states, actions=None):
        # obtain actor output which is policy action u(s_t)
        tmp_actions = np.zeros((states.shape[0], self.action_dim))
        workspace.FeedBlob(self.X_state, states.astype(np.float32))
        workspace.FeedBlob(self.X_action, tmp_actions.astype(np.float32))
        workspace.RunNet(self.q_model_test.Proto().name)
        action_actor = workspace.FetchBlob(self.actor_q_output)

        # obtain Q val for current action a_t otherwise policy u(s_t)
        used_actions = actions if actions is not None else action_actor
        workspace.FeedBlob(self.X_action, used_actions.astype(np.float32))
        workspace.RunNet(self.q_model_test.net.Proto().name)
        q_values_critics = workspace.FetchBlob(self.critic_q_output)

        return q_values_critics, action_actor

    def train_rl_nn(self, states, actions, q_vals_target):
        workspace.FeedBlob(self.X_state, states)
        workspace.FeedBlob(self.X_action, actions)
        workspace.FeedBlob(self.Y_qval, q_vals_target.astype(np.float32))
        workspace.RunNet(self.q_model_train.Proto().name)
        loss = [workspace.FetchBlob(l) for l in self.q_model_train_loss]

        loss = loss[0]
        return loss

    # network utility fucnctions for eval and train
    def get_q_val_action(self, states, actions):
        q_values_actions, _ = self.eval_rl_nn(states, actions)
        return q_values_actions

    def get_q_val_best(self, states, all_possible_actions=None):
        q_values_max, best_policy = self.eval_rl_nn(states)
        return q_values_max

    def get_action_policy_batch(self, states, all_possible_actions=None):
        q_vals, action_actor = self.eval_rl_nn(states)
        q_values_critics = workspace.FetchBlob(self.critic_q_output)
        q_values_all = q_values_critics  # not exist for actor-critics
        return action_actor, q_values_critics, q_values_all

    def get_action_policy_single(self, state, policy_test=False):
        states = np.array([state]).astype(np.float32)
        act_all, q_values_max, q_values_original = self.get_action_policy_batch(
            states)
        action = act_all[0]
        if not policy_test:
            action = deterministic_noise(action, self.noise_sigma)
        qval_a = q_values_max[0]
        qval_all = q_values_original[0]
        return action.flatten(), qval_a, qval_all
