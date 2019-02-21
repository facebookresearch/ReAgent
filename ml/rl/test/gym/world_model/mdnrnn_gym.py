#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import argparse
import json
import logging
import sys

import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core
from ml.rl.models.mdn_rnn import MDNRNNMemoryPool
from ml.rl.models.world_model import MemoryNetwork
from ml.rl.test.gym.open_ai_gym_environment import OpenAIGymEnvironment
from ml.rl.test.gym.run_gym import dict_to_np
from ml.rl.thrift.core.ttypes import MDNRNNParameters
from ml.rl.training.world_model.mdnrnn_trainer import MDNRNNTrainer


logger = logging.getLogger(__name__)

USE_CPU = -1


def get_replay_buffer(num_episodes, seq_len, max_step, gym_env):
    num_transitions = num_episodes * max_step
    samples = gym_env.generate_random_samples(
        num_transitions=num_transitions,
        use_continuous_action=True,
        max_step=max_step,
        multi_steps=seq_len,
    )

    replay_buffer = MDNRNNMemoryPool(max_replay_memory_size=num_transitions)
    # convert RL sample format to MDN-RNN sample format
    transition_terminal_index = [-1]
    for i in range(1, len(samples.mdp_ids)):
        if samples.terminals[i][0] is True:
            assert len(samples.terminals[i]) == 1
            transition_terminal_index.append(i)

    for i in range(len(transition_terminal_index) - 1):
        episode_start = transition_terminal_index[i] + 1
        episode_end = transition_terminal_index[i + 1]

        for j in range(episode_start, episode_end + 1):
            if len(samples.terminals[j]) != seq_len:
                continue
            state = dict_to_np(
                samples.states[j], np_size=gym_env.state_dim, key_offset=0
            )
            action = dict_to_np(
                samples.actions[j],
                np_size=gym_env.action_dim,
                key_offset=gym_env.state_dim,
            )
            next_actions = np.float32(
                [
                    dict_to_np(
                        samples.next_actions[j][k],
                        np_size=gym_env.action_dim,
                        key_offset=gym_env.state_dim,
                    )
                    for k in range(seq_len)
                ]
            )
            next_states = np.float32(
                [
                    dict_to_np(
                        samples.next_states[j][k],
                        np_size=gym_env.state_dim,
                        key_offset=0,
                    )
                    for k in range(seq_len)
                ]
            )
            rewards = np.float32(samples.rewards[j])
            terminals = np.float32(samples.terminals[j])
            not_terminals = np.logical_not(terminals)
            mdnrnn_state = np.vstack((state, next_states))[:-1]
            mdnrnn_action = np.vstack((action, next_actions))[:-1]

            assert mdnrnn_state.shape == (seq_len, gym_env.state_dim)
            assert mdnrnn_action.shape == (seq_len, gym_env.action_dim)
            assert rewards.shape == (seq_len,)
            assert next_states.shape == (seq_len, gym_env.state_dim)
            assert not_terminals.shape == (seq_len,)

            replay_buffer.insert_into_memory(
                mdnrnn_state, mdnrnn_action, next_states, rewards, not_terminals
            )

    return replay_buffer


def main(args):
    parser = argparse.ArgumentParser(
        description="Train a Mixture-Density-Network RNN net to learn an OpenAI"
        " Gym environment, i.e., predict next state, reward, and"
        " terminal signal using current state and action"
    )
    parser.add_argument("-p", "--parameters", help="Path to JSON parameters file.")
    parser.add_argument(
        "-g",
        "--gpu_id",
        help="If set, will use GPU with specified ID. Otherwise will use CPU.",
        default=USE_CPU,
    )
    parser.add_argument(
        "-l",
        "--log_level",
        help="If set, use logging level specified (debug, info, warning, error, "
        "critical). Else defaults to info.",
        default="info",
    )
    args = parser.parse_args(args)

    if args.log_level not in ("debug", "info", "warning", "error", "critical"):
        raise Exception("Logging level {} not valid level.".format(args.log_level))
    else:
        logger.setLevel(getattr(logging, args.log_level.upper()))

    with open(args.parameters, "r") as f:
        params = json.load(f)

    return mdnrnn_gym(params, args.gpu_id)


def mdnrnn_gym(params, gpu_id):
    logger.info("Running gym with params")
    logger.info(params)

    env_type = params["env"]
    env = OpenAIGymEnvironment(env_type, epsilon=1.0, softmax_policy=True, gamma=0.99)

    use_gpu = gpu_id != USE_CPU
    if use_gpu:
        raise NotImplementedError()

    trainer = create_trainer(params, env)
    c2_device = core.DeviceOption(
        caffe2_pb2.CUDA if use_gpu else caffe2_pb2.CPU, int(gpu_id)
    )
    return train_sgd(
        c2_device,
        env,
        trainer,
        "{} test run".format(env_type),
        params["mdnrnn"]["minibatch_size"],
        **params["run_details"],
    )


def train_sgd(
    c2_device,
    gym_env,
    trainer,
    test_run_name,
    minibatch_size,
    seq_len=5,
    num_train_episodes=300,
    num_test_episodes=100,
    max_steps=None,
    train_epochs=100,
    early_stopping_patience=3,
):
    train_replay_buffer = get_replay_buffer(
        num_train_episodes, seq_len, max_steps, gym_env
    )
    valid_replay_buffer = get_replay_buffer(
        num_test_episodes, seq_len, max_steps, gym_env
    )
    test_replay_buffer = get_replay_buffer(
        num_test_episodes, seq_len, max_steps, gym_env
    )
    valid_loss_history = []

    num_batch_per_epoch = train_replay_buffer.memory_size // minibatch_size
    logger.info(
        "Collected data {} transitions.\n"
        "Training will take {} epochs, with each epoch having {} mini-batches"
        " and each mini-batch having {} samples".format(
            train_replay_buffer.memory_size,
            train_epochs,
            num_batch_per_epoch,
            minibatch_size,
        )
    )

    for i_epoch in range(train_epochs):
        for i_batch in range(num_batch_per_epoch):
            training_batch = train_replay_buffer.sample_memories(minibatch_size)
            losses = trainer.train(training_batch)
            logger.info(
                "{}-th epoch, {}-th minibatch: \n"
                "loss={}, bce={}, gmm={}, mse={} \n"
                "cum loss={}, cum bce={}, cum gmm={}, cum mse={}\n".format(
                    i_epoch,
                    i_batch,
                    losses["loss"],
                    losses["bce"],
                    losses["gmm"],
                    losses["mse"],
                    np.mean(trainer.cum_loss),
                    np.mean(trainer.cum_bce),
                    np.mean(trainer.cum_gmm),
                    np.mean(trainer.cum_mse),
                )
            )
        # earlystopping
        trainer.mdnrnn.mdnrnn.eval()
        valid_batch = valid_replay_buffer.sample_memories(
            valid_replay_buffer.memory_size
        )
        valid_losses = trainer.get_loss(valid_batch, state_dim=gym_env.state_dim)
        valid_loss_history.append(valid_losses)
        trainer.mdnrnn.mdnrnn.train()
        logger.info(
            "{}-th epoch, validate loss={}, bce={}, gmm={}, mse={}".format(
                i_epoch,
                valid_losses["loss"],
                valid_losses["bce"],
                valid_losses["gmm"],
                valid_losses["mse"],
            )
        )
        latest_loss = valid_loss_history[-1]["loss"]
        recent_valid_loss_hist = valid_loss_history[-1 - early_stopping_patience : -1]
        if len(valid_loss_history) > early_stopping_patience and all(
            [latest_loss >= v["loss"] for v in recent_valid_loss_hist]
        ):
            break

    trainer.mdnrnn.mdnrnn.eval()
    test_batch = test_replay_buffer.sample_memories(test_replay_buffer.memory_size)
    test_losses = trainer.get_loss(test_batch, state_dim=gym_env.state_dim)
    logger.info(
        "Test loss: {}, bce={}, gmm={}, mse={}".format(
            test_losses["loss"],
            test_losses["bce"],
            test_losses["gmm"],
            test_losses["mse"],
        )
    )
    logger.info("Valid loss history: {}".format(valid_loss_history))
    return test_losses, valid_loss_history, trainer


def create_trainer(params, env):
    mdnrnn_params = MDNRNNParameters(**params["mdnrnn"])
    mdnrnn_net = MemoryNetwork(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        num_hiddens=mdnrnn_params.hidden_size,
        num_hidden_layers=mdnrnn_params.num_hidden_layers,
        num_gaussians=mdnrnn_params.num_gaussians,
    )
    cum_loss_hist_len = (
        params["run_details"]["num_train_episodes"]
        * params["run_details"]["max_steps"]
        // mdnrnn_params.minibatch_size
    )
    trainer = MDNRNNTrainer(
        mdnrnn_network=mdnrnn_net, params=mdnrnn_params, cum_loss_hist=cum_loss_hist_len
    )
    return trainer


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = sys.argv
    main(args[1:])
