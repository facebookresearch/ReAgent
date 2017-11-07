'''
Copyright (c) 2017-present, Facebook, Inc.
All rights reserved.

 This source code is licensed under the BSD-style license found in the
 LICENSE file in the root directory of this source tree. An additional grant
 of patent rights can be found in the PATENTS file in the same directory.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import deque
import json
import os
import numpy as np

import gym
from gym.spaces import Discrete, Box
from gym import wrappers

from ml.rl.test.gym.rlmodels.rlmodel_helper import sample_memories,\
    MODEL_T, one_hot, get_session_id, MODEL_PATH, MONITOR_FOLDER

from ml.rl.test.gym.rlmodels.dqn import DQN_rlnn
from ml.rl.test.gym.rlmodels.actor_critic import ActorCritic_rlnn


def normalize(x, low_high):
    """Normalize a vector x to the range [low, high]."""
    if low_high[0] == -np.finfo(np.float32).max or \
            low_high[1] == -np.finfo(np.float32).max or \
            (low_high[0] == -1 and low_high[1] == 1):
        return x
    return 2 * (x - low_high[0]) / float(low_high[1] - low_high[0]) - 1


def denormalize(x, low_high):
    """Inverse-normalize a vector x to the range [low, high]."""
    if low_high[0] == -np.finfo(np.float32).max or \
            low_high[1] == -np.finfo(np.float32).max or \
            (low_high[0] == -1 and low_high[1] == 1):
        return x
    return (x + 1) * (low_high[1] - low_high[0]) / 2.0 + low_high[0]


def identify_env_input(env_input):
    """Identify and return the functions to preprocess and recover an input."""
    env_input_type = str(env_input)
    env_input_shape = None
    env_input_process = None
    env_input_recover = None
    env_input_range = None
    image_rgb_or_bw = True

    # common type : Discrete, Box, Tuple, MultiDiscrete, MultiBinary
    if isinstance(env_input, Discrete):
        env_input_shape = (env_input.n, )  # one_hot discrete

        def env_input_process(a):
            return one_hot(a, env_input.n)

        def env_input_recover(a):
            return np.argmax(a)
    elif isinstance(env_input, Box):
        env_input_shape = env_input.shape
        # checking if it is image input type
        input_img = len(env_input.shape) == 3 and env_input.shape[-1] == 3
        if input_img:
            env_input_type = 'IMG'
            env_input_range = list(zip(env_input.low, env_input.high))[0]
        if input_img and image_rgb_or_bw:
            env_input_shape = env_input.shape

            def env_input_process(a):
                return np.rollaxis(a, 2) * (1.0 / 128.0) - 1.0

            def env_input_recover(a):
                return (np.rollaxis(a, 0, 3) + 1.0) * 128.0
        elif input_img and not image_rgb_or_bw:
            env_input_shape = (env_input.shape[0], env_input.shape[1], 1)

            def env_input_process(a):
                return np.expand_dims(
                    np.matmul(a, np.array([0.299, 0.587, 0.114])), 0
                ) * (1.0 / 128.0) - 1.0

            def env_input_recover(a):
                return np.title((a + 1.0) * 128.0, (3, 1, 1))
        else:
            # regular continuous, need flatten()
            env_input_shape = (np.prod(env_input_shape), )
            env_input_range = list(zip(env_input.low, env_input.high))

            def env_input_process(a):
                return np.array(
                    [
                        normalize(a[i], env_input_range[i])
                        for i in range(len(a.flatten()))
                    ]
                )

            def env_input_recover(a):
                return np.array(
                    [
                        denormalize(a[i], env_input_range[i])
                        for i in range(len(a.flatten()))
                    ]
                )
    else:
        raise ValueError("Unknown space type: ", env_input)
    return env_input_shape, env_input_type, env_input_range,\
        env_input_process, env_input_recover


def Setup(args):
    if args.gymenv in [e.id for e in gym.envs.registry.all()]:
        env = gym.make(args.gymenv)
        print("Env gym: ", args.gymenv)
    else:
        print(
            "Warning: Env {} not fount in openai gym, quit.".
            format(args.gymenv)
        )
        exit()

    # if actually render env
    if args.render:
        proposed_MONITOR_FOLDER = MONITOR_FOLDER + get_session_id(args)
        if not os.path.isdir(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        if not os.path.isdir(MONITOR_FOLDER):
            os.mkdir(MONITOR_FOLDER)
        if os.path.isdir(proposed_MONITOR_FOLDER):
            print(
                "Warning: monitor output folder {} exists, overwriting".
                format(proposed_MONITOR_FOLDER)
            )
        else:
            os.mkdir(proposed_MONITOR_FOLDER)
        # overwriting
        env = wrappers.Monitor(env, proposed_MONITOR_FOLDER, force=True)

    state_shape, state_type, state_range, _, _ = identify_env_input(
        env.observation_space
    )
    action_shape, _, action_range, _, _ = identify_env_input(env.action_space)

    print(
        "Env setting: state/action type(shape):", env.observation_space,
        env.action_space
    )

    return env, state_shape, state_type, action_shape, action_range


def TrainModel(rlnn, env, args):
    """Train an RL model."""
    if rlnn is None:
        print("Model not initalized properly, force quit")
        exit()

    _, _, _, state_proc, _ = \
        identify_env_input(env.observation_space)
    _, _, _, _, action_recv =\
        identify_env_input(env.action_space)

    onlyTest = args.test
    renderGym = args.render
    save_path = args.path

    n_steps = args.number_steps_total
    n_steps_timeout = args.number_steps_timeout
    n_iterations = args.number_iterations
    learning_batch_num_every_iteration = args.learn_batch_num_every_iteration
    learning_every_n_iterations = args.learn_every_n_iterations
    save_every_iterations = args.save_iteration
    batch_size = args.batch_size

    session_id = get_session_id(args)

    learning_start_iteration = 10
    render_every_iterations = 100

    # Replay memory, epsilon-greedy policy and observation preprocessing
    replay_memory_size = 100000
    replay_memory = deque([], maxlen=replay_memory_size)
    avg_reward_queue_size = 1000
    avg_reward_queue = deque([], maxlen=avg_reward_queue_size)
    replay_memory.clear()
    avg_reward_queue.clear()

    # Execution phase
    step = 0
    global_step = 0
    survival_step = 0

    # reward avg to track performance
    reward_per_iter = 0
    terminal = True

    avg_reward_iters = []
    avg_reward_records = []
    loss_iters = []
    loss_records = []
    reward_test_every_iter = 10 if not onlyTest else 1
    reward_count_dump_iter = 100
    reward_avg_over_iter = reward_count_dump_iter // reward_test_every_iter

    state = None
    next_state = None
    action = None
    next_action = None
    reward = None

    training_testing_session = "TESTING"
    if not onlyTest:
        training_testing_session = "TRAINING"
        print(
            "Training: Train every {} iteration, for {} batches, with batch_size {}".
            format(
                learning_every_n_iterations, learning_batch_num_every_iteration,
                batch_size
            )
        )
        print(
            "Training: Reward collected every {}, " +
            "avg computed every {} over {} test trial".format(
                reward_count_dump_iter, reward_test_every_iter,
                reward_avg_over_iter
            )
        )
    print("\n=== {} START ====\n".format(training_testing_session))

    iteration = 0
    while iteration < n_iterations:
        if global_step > n_steps and n_steps > 0:
            break

        if iteration % reward_test_every_iter == 0:
            avg_reward_queue.append(reward_per_iter)
        if iteration % reward_count_dump_iter == 0:
            last_avg_reward = list(avg_reward_queue)[-reward_avg_over_iter:]
            last_avg_reward = np.mean(np.array(last_avg_reward))
            print(
                '\rIter {}\t Avg Reward: {}\t'.
                format(iteration, last_avg_reward)
            )

            avg_reward_iters.append(iteration)
            avg_reward_records.append(round(last_avg_reward, 2))

        step = 0
        survival_step = 0
        reward_per_iter = 0

        iteration += 1
        test_iter = iteration % reward_test_every_iter == 0

        terminal = False
        obs = env.reset()
        next_state = state_proc(obs)
        next_action = rlnn.get_policy(next_state, test_iter)

        while not terminal:
            state = next_state
            action = next_action
            obs, reward, terminal, info = env.step(action_recv(action))
            next_state = state_proc(obs)
            next_action = rlnn.get_policy(next_state, test_iter)

            if n_steps_timeout > 0 and step > n_steps_timeout:
                terminal = True
                break

            step += 1
            survival_step += 1
            global_step += 1
            reward_per_iter += reward

            # memorize session in replay memory
            replay_memory.append(
                (
                    state.astype(np.float32), action.astype(np.float32), reward,
                    terminal, next_state.astype(np.float32),
                    next_action.astype(np.float32)
                )
            )

            if renderGym and iteration % render_every_iterations == 0:
                env.render()

        # if only test, we skip the replay memory and training
        if onlyTest:
            continue

        # else training using batches of samples
        if iteration > learning_start_iteration and\
                iteration % learning_every_n_iterations == 0:
            loss_curr = []
            for _ in range(learning_batch_num_every_iteration):
                batch_samples = sample_memories(replay_memory, batch_size)
                if (
                    rlnn.model_type == MODEL_T.DQN_ADAPTED.name or
                    rlnn.model_type == MODEL_T.SARSA_ADAPTED.name
                ):
                    loss = rlnn.train(*batch_samples)
                else:
                    loss, _ = rlnn.train(*batch_samples)
                loss_curr.append(loss)

            loss_iters.append(iteration)
            loss_records.append(np.mean(np.array(loss_curr)))

        if args.nosave is False and save_every_iterations > 0 and \
                iteration % save_every_iterations == 0:
            SaveModel(rlnn, save_path, session_id)  # + str(iteration)

    avg_reward_tracking = list(zip(avg_reward_iters, avg_reward_records))
    print("\n=== {} FINISHED ====\n".format(training_testing_session))
    print("Summary: (iter, reward) =", avg_reward_tracking)

    if args.nosave is False and not onlyTest:
        SaveModel(rlnn, save_path, session_id)

    return avg_reward_tracking


# Saving and loading models


def SaveModel(rlnn, MODEL_PATH, session_id):
    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    rlnn.save_args(MODEL_PATH, session_id)
    rlnn.save_nn_params(MODEL_PATH, session_id)
    return


def LoadModel(MODEL_PATH, session_id):
    args_load = []
    with open(os.path.join(MODEL_PATH, session_id + "_args.txt"), 'r') as fid:
        args_load = json.loads(fid.read())
        fid.close()
    print("Model loading: args=", args_load)
    if len(args_load) < 2:
        print("Warning: parameters for rlnn incomplete. Exit.")
        exit()
    rlnn = None
    model_type = args_load['model_type']
    if model_type == MODEL_T.DQN.name:
        rlnn = DQN_rlnn(**args_load)
    elif model_type == MODEL_T.ACTORCRITIC.name:
        rlnn = ActorCritic_rlnn(**args_load)

    rlnn.load_nn_params(MODEL_PATH, session_id)
    return rlnn
