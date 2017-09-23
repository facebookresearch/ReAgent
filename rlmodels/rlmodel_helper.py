'''
Copyright (c) 2017-present, Facebook, Inc.
All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
'''

# @package rlmodel_helper
# Module rl2_caffe2.rlmodels.rlmodel_helper

from caffe2.python.optimizer import (
    build_sgd, build_ftrl, build_adagrad, build_adam)
from caffe2.python import brew


import numpy as np
from enum import Enum
import os

# define helper function for training

#  policy: * $\epsilon$-greedy for discrete action:
#  a_t = u(s_t) with (1-$\epsilon$) chance, otherwise rand
DEFAULT_EPSILON = 0.2

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = BASE_PATH + '/outputs/'
MONITOR_FOLDER = BASE_PATH + '/outputs/saved_sessions/'  # for openai-gym upload


def get_session_id(args):
    """Get a unique id for the current session"""
    return "_".join([args.gymenv, args.model_type, args.model_id])


def epsilon_greedy(action_discrete, n_action_outputs, epsilon=DEFAULT_EPSILON):
    if np.random.rand() < epsilon:
        return np.random.randint(n_action_outputs)  # random action
    else:
        return action_discrete


def epsilon_greedy_onehot(action_discrete, n_action_outputs, epsilon=DEFAULT_EPSILON):
    if np.random.rand() < epsilon:
        act = np.random.randint(n_action_outputs)  # random action
        return one_hot(act, n_action_outputs)
    else:
        return action_discrete

# policy * deterministic noise policy for continuous action:
# $a_t = u(s_t) + noise = u(s_t) + N(\mu,\sigma)$


def deterministic_noise(action_continous, sigma=0, mu=0):
    noise = mu + sigma * np.random.rand(*action_continous.shape)
    return action_continous + noise

# other helper function
# one hot vector for wrapping inputs


def one_hot(n, vec_dim):
    vector_onehot = np.zeros(vec_dim, )
    vector_onehot[n] = 1
    return vector_onehot

# model type


class MODEL_T(Enum):
    SARSA = 0
    DQN = 1
    ACTORCRITIC = 2


MODEL_T_DICT = dict([(op.name, op) for op in MODEL_T])

# training optimizer wrappers


class GRAD_OPTIMIZER(Enum):
    SGD = 0
    STEP = 1
    MOMENTUM = 2
    RMSP = 3
    ADAGRAD = 4
    ADAM = 5
    FTRL = 6


OPTIMIZER_DICT = dict([(op.name, op) for op in GRAD_OPTIMIZER])
DEFAULT_OPTIMIZER = "SGD"

def add_parameter_update_ops(model, optimizer_input="SGD",
                             base_learning_rate=0.01):
    optimizer_rule = OPTIMIZER_DICT[optimizer_input]
    optimizer_here = None
    if optimizer_rule == GRAD_OPTIMIZER.SGD:
        optimizer_here = build_sgd(model, base_learning_rate)
    elif optimizer_rule == GRAD_OPTIMIZER.ADAGRAD:
        optimizer_here = build_adagrad(model, base_learning_rate)
    elif optimizer_rule == GRAD_OPTIMIZER.ADAM:
        optimizer_here = build_adam(model, base_learning_rate)
    elif optimizer_rule == GRAD_OPTIMIZER.FTRL:
        optimizer_here = build_ftrl(model, base_learning_rate)
    else:
        add_parameter_update_ops_others(model, optimizer_input, base_learning_rate)
    return

def add_parameter_update_ops_others(model, optimizer_input='SGD',
                                base_learning_rate=0.01):
    optimizer_rule = OPTIMIZER_DICT[optimizer_input]
    iter = brew.iter(model, "iter")
    lr_policy = "step" if optimizer_rule == GRAD_OPTIMIZER.STEP else "fixed"
    lr = model.net.LearningRate(
        [iter],
        "lr" ,
        base_lr=base_learning_rate,
        policy=lr_policy
    )
    if optimizer_rule == GRAD_OPTIMIZER.STEP:
        for param in model.GetParams():
            if param in model.param_to_grad:
                param_grad = model.param_to_grad[param]
                model.net.Mul([param_grad, lr], param_grad+'_lrscaled', broadcast=1)
                model.net.Sub([param, param_grad+'_lrscaled'], param)
    elif optimizer_rule == GRAD_OPTIMIZER.MOMENTUM:
        for param in model.GetParams():
            if param in model.param_to_grad:
                param_grad = model.param_to_grad[param]
                param_momentum = model.param_init_net.ConstantFill(
                    [param], param + '_momentum', value=0.0
                )
                # Update param_grad and param_momentum in place
                model.net.MomentumSGDUpdate(
                    [param_grad, param_momentum, lr, param],
                    [param_grad, param_momentum, param],
                    momentum=0.9,
                    nesterov=1,
                )
                model.net.Mul([param_grad, lr], [param_grad+'_lrscaled'], broadcast=1)
                model.net.Sub([param, param_grad+'_lrscaled'], param)
            else:
                print("WARNING: param does not has gradient", param)
    elif optimizer_rule == GRAD_OPTIMIZER.RMSP:
        for param in model.GetParams():
            if param in model.param_to_grad:
                param_grad = model.param_to_grad[param]
                param_momentum = model.param_init_net.ConstantFill(
                    [param], param + '_momentum', value=0.0
                )
                param_meansq = model.param_init_net.ConstantFill(
                    [param], param + '_meansq', value=0.0
                )
                model.net.RmsProp(
                    [param_grad, param_meansq, param_momentum, lr],
                    [param_grad, param_meansq, param_momentum],

                )
                model.net.Mul([param_grad, lr], param_grad+'_lrscaled', broadcast=1)
                model.net.Sub([param, param_grad+'_lrscaled'], param)
            else:
                print("WARNING: param does not has gradient", param)
    else:
        print("unrecognized in caffe2 setting, using default SGD",
              optimizer_rule)
        optimizer_here = build_sgd(model, base_learning_rate)
    return

# replay memory sample


def sample_memories(replay_memory, batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    # state, action, reward, terminal, next_state, next_action
    cols = [[], [], [], [], [], []]
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3].reshape(-1, 1), cols[4], cols[5]
