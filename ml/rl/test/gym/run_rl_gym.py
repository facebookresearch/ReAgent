'''
Copyright (c) 2017-present, Facebook, Inc.
All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
'''

# coding: utf-8
# @package reinforcement_learning_caffe2
# Module caffe2.python.examples.reinforcement_learning

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys

import gym

from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2

from ml.rl.test.gym.rlmodels.rlmodel_helper import MODEL_T, MODEL_T_DICT, \
    get_session_id, MODEL_PATH, MONITOR_FOLDER
from ml.rl.training.model_update_helper import GRAD_OPTIMIZER, OPTIMIZER_DICT

from ml.rl.test.gym.rlmodels.dqn import DQN_rlnn
from ml.rl.test.gym.rlmodels.dqn_adapted import DQN_rlnn_Adapted
from ml.rl.test.gym.rlmodels.actor_critic import ActorCritic_rlnn
from ml.rl.test.gym.trainer import Setup, TrainModel, LoadModel

# # RL Experiments on OpenAI Gym
# ## Experiment: RL algorithms on OpenAI Gym benchmarks
#
# ### Dependencies:
# 1. caffe2, numpy
# 2. gym (note: only gym-basic is installed in non-opt mode, which includes:
#   [algorithmic](https://gym.openai.com/envs#algorithmic),
#   [toy_text](https://gym.openai.com/envs#toy_text)
#   [classic_control](https://gym.openai.com/envs#classic_control)


def upload_to_openai(session_id):
    """Publish model to OpenAI Gym"""
    gym.scoreboard.api_key = ''
    if gym.scoreboard.api_key == '':
        raise ValueError("Please put your OpenAI Gym Key in run_rl_gym.py")
    gym.upload(MONITOR_FOLDER + session_id, api_key=gym.scoreboard.api_key)


def Run(args):
    session_id = get_session_id(args)
    env, state_shape, state_type, action_shape, action_range = Setup(args)
    action_constraints = args.constraint
    input_img = 'IMG' in state_type

    model_id = args.model_id
    model_type = MODEL_T_DICT[args.model_type]\
        if args.model_type in MODEL_T_DICT else MODEL_T.DQN
    print("Model Id: ", model_id)
    print("Model Type: ", model_type)

    maxq_learning = False if model_type == MODEL_T.SARSA_ADAPTED else True

    if args.test:
        save_path = args.path or MODEL_PATH
        rlnn = LoadModel(save_path, session_id)
        assert (rlnn is not None)
        assert (args.test)
        TrainModel(rlnn, env, args)
    else:
        # init and train model
        learning_rate = args.learning_rate
        discount_gamma = args.discount_gamma
        optimizer = args.optimizer if args.optimizer in OPTIMIZER_DICT \
            else GRAD_OPTIMIZER.SGD
        print("Model Optimizer: ", optimizer)

        # hidden neural network
        if model_type in [MODEL_T.DQN, MODEL_T.DQN_ADAPTED, MODEL_T.SARSA_ADAPTED]:
            n_hidden_nodes = [16]
            n_hidden_activations = ['relu']
        if model_type == MODEL_T.ACTORCRITIC:
            n_hidden_nodes = [256, 64, 32]
            n_hidden_activations = ['relu', 'relu', 'linear']
        print("Model NN layers:", n_hidden_nodes, n_hidden_activations)

        rlnn = None
        if model_type == MODEL_T.DQN:
            rlnn = DQN_rlnn(
                model_id,
                model_type.name,
                state_shape,
                action_shape,
                discount_gamma,
                learning_rate,
                optimizer,
                n_hidden_nodes,
                n_hidden_activations,
                maxq_learning=maxq_learning,
                input_img=input_img
            )
        elif model_type == MODEL_T.DQN_ADAPTED or model_type == MODEL_T.SARSA_ADAPTED:
            rlnn = DQN_rlnn_Adapted(
                model_id,
                model_type.name,
                state_shape,
                action_shape,
                discount_gamma,
                learning_rate,
                optimizer,
                n_hidden_nodes,
                n_hidden_activations,
                maxq_learning=maxq_learning
            )
        elif model_type == MODEL_T.ACTORCRITIC:
            action_constraints = True
            rlnn = ActorCritic_rlnn(
                model_id,
                model_type.name,
                state_shape,
                action_shape,
                discount_gamma,
                learning_rate,
                optimizer,
                n_hidden_nodes,
                n_hidden_activations,
                action_constraints=action_constraints,
                maxq_learning=maxq_learning,
                input_img=input_img
            )

        return TrainModel(rlnn, env, args)

    if args.render and args.upload:
        upload_to_openai(session_id)

    print("\nfinished")


def main(argv_new):
    """Main entrypoint"""
    parser = argparse.ArgumentParser(
        description="Train a RL net to play in openAI GYM."
    )
    parser.add_argument(
        "-x",
        "--number-steps-total",
        type=int,
        help="total number of training steps",
        default=1000000
    )
    parser.add_argument(
        "-w",
        "--number-steps-timeout",
        type=int,
        help="number of steps before time out",
        default=-1
    )
    parser.add_argument(
        "-i",
        "--number-iterations",
        type=int,
        help="total number of iterations",
        default=1000
    )
    parser.add_argument(
        "-y",
        "--learn-every-n-iterations",
        type=int,
        help="training every n numbers of game iterations",
        default=2
    )
    parser.add_argument(
        "-z",
        "--learn-batch-num-every-iteration",
        type=int,
        help="batch number for learning each time",
        default=100
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        help="batch size for training",
        default=128
    )

    parser.add_argument(
        "-s",
        "--save-iteration",
        type=int,
        help="saving checkpoint every n number of iterations",
        default=-1
    )
    parser.add_argument(
        "-p", "--path", help="path of the checkpoint file", default=MODEL_PATH
    )
    parser.add_argument(
        "--nosave", help="Don't save the model to disk", action='store_true'
    )

    parser.add_argument(
        "-c",
        "--constraint",
        help="constrained actions",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "-t",
        "--test",
        help="test (no learning and minimal epsilon)",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "-u",
        "--upload",
        help="upload after finishing training/testing",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        help="increase output verbosity",
        default=0
    )

    parser.add_argument(
        "-g",
        "--gymenv",
        help="specify gym env for training",
        default="CartPole-v0"
    )
    parser.add_argument(
        "-r",
        "--render",
        help="render training",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "-a",
        "--model-id",
        help="specify training model unique id",
        default="new"
    )
    parser.add_argument(
        "-m",
        "--model-type",
        help="specify training model type:\
                        DQN or ACTORCRITIC",
        default="DQN"
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        help="specify optimizer for training",
        default="SGD"
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        help="specify learning rate for training",
        default=0.01
    )
    parser.add_argument(
        "-d",
        "--discount-gamma",
        type=float,
        help="specify discounted factor gamma for RL",
        default=0.9
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="If set, training is going to use GPU 0",
        default=False
    )

    args = parser.parse_args(argv_new)
    print("args:", args)

    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    workspace.ResetWorkspace()

    device = core.DeviceOption(
        caffe2_pb2.CUDA if args.gpu else caffe2_pb2.CPU, 0
    )
    with core.DeviceScope(device):
        return Run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
