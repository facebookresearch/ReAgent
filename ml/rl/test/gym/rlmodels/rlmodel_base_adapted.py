'''
Copyright (c) 2017-present, Facebook, Inc.
All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
'''

# @package rlmodel_base
# Module rl2_caffe2.rlmodels.rlmodel_base

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
from os import path

import numpy as np
from caffe2.python import workspace
from caffe2.python import net_drawer
from ml.rl.training.ml_trainer import MLTrainer
import caffe2.python.predictor.predictor_exporter as pe
from ml.rl.thrift.core.ttypes import TrainingParameters


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


# ### Base Model Class for all RL algorithm supported,
# , defining all necessary functions and interface
# 1. DQN-max-action: Deep Q Network
class RLNN_Adapted(MLTrainer):
    def __init__(
        self,
        model_id,
        model_type,
        state_shape,
        action_shape,
        gamma,
        learning_rate,
        optimizer,
        n_hidden_nodes,
        n_hidden_activations,
        maxq_learning=True
    ):
        self.saving_args = {k: v for k, v in locals().items() if k != 'self'}
        self.model_type = model_type
        self.input_dim = state_shape[0]
        self.output_dim = action_shape[0]
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_activations = n_hidden_activations
        self.maxq_learning = maxq_learning

        layers = [self.input_dim] + self.n_hidden_nodes + [self.output_dim]
        activations = self.n_hidden_activations + ['linear']

        parameters = TrainingParameters(
            layers=layers,
            activations=activations,
            minibatch_size=None,
            optimizer=optimizer,
        )

        MLTrainer.__init__(self, model_id, parameters)

    def _setup_initial_blobs(self):
        self.init_params()
        self.action_blob = "action"
        workspace.FeedBlob(self.action_blob, np.zeros(1, dtype=np.float32))

        MLTrainer._setup_initial_blobs(self)

    def init_params(self):
        # this might be different for specific networks
        return

    def construct_rl_nn(self, q_model):
        raise NotImplementedError("Please Implement this method")

    # network utility fucnctions for eval and train
    def run_score_rl_nn(self, states, actions=None):
        raise NotImplementedError("Please Implement this method")

    def run_train_rl_nn(self, states, actions, q_vals_target):
        raise NotImplementedError("Please Implement this method")

    def get_q_val_action(self, states, actions):
        raise NotImplementedError("Please Implement this method")

    def get_q_val_best(self, states):
        raise NotImplementedError("Please Implement this method")

    def get_policy(self, state, policy_test=False):
        raise NotImplementedError("Please Implement this method")

    def train(
        self,
        states,
        actions,
        rewards,
        terminals,
        next_states,
        next_actions,
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
            action: actions[i][j] = 1 if action_i == j else 0
        :param rewards: Numpy array with shape (batch_size, 1). The ith entry
            is the reward experienced at the ith transition.
        :param terminals: Numpy array with shape (batch_size, 1). The ith entry
            is equal to 1 iff the ith transition's next state was terminal.
        :param next_states: Numpy array with shape (batch_size, state_dim). The
            ith row is a representation of the ith transition's next state.
        :param next_actions: Numpy array with shape (batch_size, action_dim). The
            ith row contains the one-hotted representation of the ith
            transition's next action: next_actions[i][j] = 1 iff next_action_i == j
        """
        if self.maxq_learning:
            next_q_values = self.get_q_val_best(next_states)
        else:
            next_q_values = self.get_q_val_action(next_states, next_actions)

        q_vals_target = rewards + (1.0 - terminals) * self.gamma * next_q_values
        self.run_train_rl_nn(states, actions, q_vals_target)

        return workspace.FetchBlob(self.loss_blob)

    def save_nn_structure(self, saving_folder, session_id):
        with open(path.join(saving_folder, session_id + "_train.pbtxt"),
                  'w') as fid:
            fid.write(str(self.train_model.net.Proto()))

        with open(path.join(saving_folder, session_id + "_test.pbtxt"),
                  'w') as fid:
            fid.write(str(self.score_model.net.Proto()))

    def save_nn_params(self, saving_folder, session_id):
        save_file_path = path.join(saving_folder, session_id + "_model.minidb")
        # construct the model to be exported
        # the inputs/outputs of the model are manually specified.
        pe_meta = pe.PredictorExportMeta(
            predict_net=self.score_model.net.Proto(),
            parameters=[str(b) for b in self.score_model.params],
            inputs=[self.input_blob, self.action_blob],
            outputs=[self.output_blob],
        )
        # save the model to a file. Use minidb as the file format
        pe.save_to_db("minidb", save_file_path, pe_meta)
        print("Model saving: successfully saved parameters/weights for models.")

    def save_args(self, saving_folder, session_id):
        with open(path.join(saving_folder, session_id + "_args.txt"),
                  'w') as fid:
            fid.write(json.dumps(self.saving_args, cls=NumpyEncoder))
            fid.close()
        print("Model saving: successfully saved constructor params.")

    def load_nn_params(
        self, saving_folder, session_id, continue_training=False
    ):
        load_file_path = path.join(saving_folder, session_id + "_model.minidb")
        print("Model loading: from path {}".format(load_file_path))
        # TODO? reset the workspace, and load the predict net
        workspace.ResetWorkspace()
        predict_net = pe.prepare_prediction_net(load_file_path, "minidb")
        workspace.RunNetOnce(predict_net)
        print(
            "Model loading: workspace blobs: {}".format(str(workspace.Blobs()))
        )
        self.score_model.net = predict_net
        print("Model loading: successfully loaded constructor params.")

    def visualize_nn(self, model):
        graph = net_drawer.GetPydotGraph(model.Proto().op, rankdir="LR")
        graph.write_png(self.model_type + '_' + self.model_id + '.png')

    def visualize_nn_test(self):
        self.visualize_nn(self.score_model)

    def visualize_nn_train(self):
        self.visualize_nn(self.train_model)
