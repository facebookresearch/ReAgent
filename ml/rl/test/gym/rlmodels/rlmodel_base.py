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
from caffe2.python import model_helper, workspace, brew
from caffe2.python import net_drawer
import caffe2.python.predictor.predictor_exporter as pe


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
# 2. Actor-Critic
class RLNN(object):
    def __init__(
        self,
        model_id,
        model_type,
        state_shape,
        action_shape,
        discount_gamma,
        learning_rate,
        optimizer,
        n_hidden_nodes,
        n_hidden_activations,
        action_constraints=False,
        maxq_learning=False,
        input_img=True
    ):
        self.saving_args = {k: v for k, v in locals().items() if k != 'self'}

        self.model_id = model_id
        self.model_type = model_type

        self.state_shape = state_shape
        self.action_shape = action_shape
        self.state_dim = self.state_shape[0]
        self.action_dim = self.action_shape[0]

        self.discount_gamma = discount_gamma
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.maxq_learning = maxq_learning

        self.n_hidden_input = self.state_dim
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_activations = n_hidden_activations
        self.action_constraints = action_constraints

        self.input_Img_CNN = input_img

        self.actor_label = "actor"
        self.critic_label = "critic"
        self.actor_prefix = self.actor_label + "_q_"
        self.critic_prefix = self.critic_label + "_q_"
        self.state_output_label = "output"
        self.action_output_label = "output_a"
        self.critic_output_label = "output_c"
        self.final_output_label = "outputfinal"

        self.setup()

    def setup(self):
        self.init_params()
        self.init_inputs()

        if self.input_Img_CNN:
            self.setup_cnn_parameters()

        self.q_model_train = model_helper.ModelHelper(name="q_model_train")
        self.q_net_train = self.q_model_train.net

        self.q_model_test = model_helper.ModelHelper(
            name="q_model_test", init_params=False
        )
        self.q_net_test = self.q_model_test.net

        # Construct NN, note: here train test share same parameter as forward,
        # but train-net contains loss and grad
        self.q_model_test, self.q_model_test_outputs, _ = \
            self.construct_rl_nn(self.q_model_test)
        self.q_model_train, self.q_model_train_outputs, self.q_model_train_loss = \
            self.construct_rl_nn(self.q_model_train, True)

        workspace.RunNetOnce(self.q_model_train.param_init_net)
        workspace.CreateNet(self.q_model_train.net, overwrite=True)
        workspace.RunNetOnce(self.q_model_test.param_init_net)
        workspace.CreateNet(self.q_model_test.net)

    def init_params(self):
        # this might be different for specific networks
        return

    def init_inputs(self):
        self.X_state = "X_state"
        self.X_action = "X_action"
        self.Y_qval = "Y_qval"
        workspace.FeedBlob(
            self.X_state, np.random.randn(1, self.state_dim).astype(np.float32)
        )
        workspace.FeedBlob(
            self.X_action,
            np.random.randn(1, self.action_dim).astype(np.float32)
        )
        workspace.FeedBlob(
            self.Y_qval, np.random.randn(1, self.output_dim).astype(np.float32)
        )

        self.critic_q_output = self.critic_prefix + self.final_output_label
        self.actor_q_output = self.actor_prefix + self.final_output_label

        self.q_model_inputs = [self.X_state, self.X_action]
        self.q_model_outputs = [self.critic_q_output]

    #  network construction at network level
    # general construct a next work with input of state (or + action) ,
    # output either action(as actor),  qval for each action (eg dqn),
    # or qval for input action (critic or drrn)

    def q_network(
        self,
        model,
        prefix,
        X_input,
        X_inputdim,
        Y_outputdim,
        X_input_action=None,
        X_input_action_dim=None
    ):
        last_layer = X_input
        layer_dim_in = X_inputdim

        if self.input_Img_CNN:
            last_layer, layer_dim_in = self.setup_cnn_network(
                model, prefix, last_layer
            )

        for n_count, n_hidden in enumerate(self.n_hidden_nodes):
            curr_layer = prefix + 'fc' + str(n_count)
            brew.fc(
                model,
                last_layer,
                curr_layer,
                dim_in=layer_dim_in,
                dim_out=n_hidden
            )

            if 'relu' in self.n_hidden_activations[n_count]:
                brew.relu(model, curr_layer, curr_layer)
            # else: by default only fc linear function

            last_layer = curr_layer
            layer_dim_in = n_hidden

        layer_dim_out = Y_outputdim
        output_layer = prefix + self.state_output_label

        brew.fc(
            model,
            curr_layer,
            output_layer,
            dim_in=layer_dim_in,
            dim_out=layer_dim_out
        )
        used_outputs = output_layer
        output_layer_a = prefix + self.action_output_label
        if self.actor_label in prefix and self.action_constraints:
            # model.net.Tanh(output_layer, output_layer_a)
            model.net.Clip(output_layer, output_layer_a, min=-1.0, max=1.0)
            used_outputs = output_layer_a

        if X_input_action is not None:
            output_layer_critic = prefix + self.critic_output_label
            X_input_action_dim = X_input_action_dim or self.action_dim
            brew.fc(
                model,
                X_input_action,
                output_layer_a,
                dim_in=X_input_action_dim,
                dim_out=layer_dim_out
            )
            brew.sum(model, [output_layer, output_layer_a], output_layer_critic)
            used_outputs = output_layer_critic

        final_outputs = prefix + self.final_output_label
        model.net.NanCheck([used_outputs], [final_outputs])

        return final_outputs, model

    def construct_rl_nn(self, q_model, trainNetwork=False):
        raise NotImplementedError("Please Implement this method")

    # network utility fucnctions for eval and train
    def eval_rl_nn(self, states, actions=None):
        raise NotImplementedError("Please Implement this method")

    def train_rl_nn(self, states, actions, q_vals_target):
        raise NotImplementedError("Please Implement this method")

    def get_q_val_action(self, states, actions):
        raise NotImplementedError("Please Implement this method")

    def get_q_val_best(self, states, all_possible_actions=None):
        raise NotImplementedError("Please Implement this method")

    def get_action_policy_batch(self, states):
        raise NotImplementedError("Please Implement this method")

    def get_action_policy_single(self, state, policy_test=False):
        raise NotImplementedError("Please Implement this method")

    def get_policy(self, state, policy_test=False):
        action, _, _ = self.get_action_policy_single(state, policy_test)
        return action

    # interface utilities/functions for trainer/runner
    def train(
        self,
        states,
        actions,
        rewards,
        terminals,
        next_states,
        next_actions=None,
        next_possible_actions=None
    ):
        gamma = self.discount_gamma
        if not self.maxq_learning and next_actions is not None:
            next_q_values_action = self.get_q_val_action(
                next_states, next_actions
            )
            q_vals_next = next_q_values_action
        else:
            next_q_values_max = self.get_q_val_best(
                next_states, next_possible_actions
            )
            q_vals_next = next_q_values_max

        q_vals_target = rewards + (1.0 - terminals) * gamma * q_vals_next

        loss = self.train_rl_nn(states, actions, q_vals_target)

        return loss, q_vals_target

    def save_nn_structure(self, saving_folder, session_id):
        with open(path.join(saving_folder, session_id + "_train.pbtxt"),
                  'w') as fid:
            fid.write(str(self.q_model_train.net.Proto()))

        with open(path.join(saving_folder, session_id + "_test.pbtxt"),
                  'w') as fid:
            fid.write(str(self.q_model_test.net.Proto()))

    def save_nn_params(self, saving_folder, session_id):
        save_file_path = path.join(saving_folder, session_id + "_model.minidb")
        # construct the model to be exported
        # the inputs/outputs of the model are manually specified.
        pe_meta = pe.PredictorExportMeta(
            predict_net=self.q_model_test.net.Proto(),
            parameters=[str(b) for b in self.q_model_test.params],
            inputs=self.q_model_inputs,
            outputs=self.q_model_outputs,
        )
        # save the model to a file. Use minidb as the file format
        pe.save_to_db("minidb", save_file_path, pe_meta)
        print("Model saving: successfully saved parameters/weights for models.")

    def save_args(self, saving_folder, session_id):
        with open(path.join(saving_folder, session_id + "_args.txt"),
                  'w') as fid:
            fid.write(json.dumps(self.saving_args, cls=NumpyEncoder))
            fid.close()
        print("Model saving: successfully saved constructor parms.")

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
        self.q_model_test.net = predict_net
        print("Model loading: successfully loaded constructor params.")

    def visualize_nn(self, model):
        graph = net_drawer.GetPydotGraph(model.Proto().op, rankdir="LR")
        graph.write_png(self.model_type + '_' + self.model_id + '.png')

    def visualize_nn_test(self):
        self.visualize_nn(self.q_model_test)

    def visualize_nn_train(self):
        self.visualize_nn(self.q_model_train)

    def setup_cnn_parameters(
        self,
        conv_n_maps=None,
        conv_kernels=None,
        conv_pools_kernels_strides=None,
        conv_pools=None
    ):
        # default cnn model structure if empty or None
        self.conv_n_maps = [32, 64, 64] if conv_n_maps is None else conv_n_maps
        self.conv_kernels = [8, 4, 3] if conv_kernels is None else conv_kernels
        self.conv_pools_kernels_strides = [4, 4, 2] \
            if conv_pools_kernels_strides is None else conv_pools_kernels_strides
        self.conv_pools = ['max', 'max', 'max'] \
            if conv_pools is None else conv_pools

        self.input_channels = self.state_shape[2]
        self.input_height = self.state_shape[0]
        self.input_width = self.state_shape[1]
        self.input_frames = 1  # look back frames, only for video input

    def setup_cnn_network(self, model, prefix, input_layer):
        # assume data is already scaled and croped and bw
        layer_count = 1
        last_layer = input_layer
        layer_dim_in = self.input_channels
        final_height = self.input_height
        final_width = self.input_width

        for n_maps, kernel_size, stride, pools in zip(
            self.conv_n_maps, self.conv_kernels,
            self.conv_pools_kernels_strides, self.conv_pools
        ):
            curr_layer = prefix + 'conv' + str(layer_count)
            brew.conv(
                model,
                last_layer,
                curr_layer,
                dim_in=layer_dim_in,
                dim_out=n_maps,
                kernel=kernel_size
            )  # order='NCHW'

            curr_layer_pool = prefix + pools + 'pool' + str(layer_count)
            if 'max' in pools:
                brew.max_pool(
                    model,
                    curr_layer,
                    curr_layer_pool,
                    kernel=stride,
                    stride=stride
                )
            elif 'avg' in pools:
                brew.average_pool(
                    model,
                    curr_layer,
                    curr_layer_pool,
                    kernel=stride,
                    stride=stride
                )
            else:
                raise Exception(
                    "CNN Pooling type not supported {}".format(pools)
                )

            layer_count += 1
            layer_dim_in = n_maps
            last_layer = curr_layer_pool

            final_height = (final_height - kernel_size + 1) // stride
            final_width = (final_width - kernel_size + 1) // stride

        cnn_output_dim = self.conv_n_maps[-1] * final_height * final_width
        print(
            "cnn final shape: {}x{}x{} = {} ".format(
                str(self.conv_n_maps[-1]),
                str(final_height), str(final_width), str(cnn_output_dim)
            )
        )
        return last_layer, cnn_output_dim
