#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import argparse
import functools
import json
import logging
import sys
from collections import deque

import ml.rl.types as rlt
import numpy as np
import torch
from ml.rl.test.base.utils import only_continuous_normalizer
from ml.rl.test.gym.open_ai_gym_environment import EnvType, OpenAIGymEnvironment
from ml.rl.test.gym.open_ai_gym_memory_pool import OpenAIGymMemoryPool
from ml.rl.test.gym.run_gym import (
    USE_CPU,
    create_epsilon,
    create_predictor,
    create_trainer,
    train_gym_offline_rl,
)
from ml.rl.test.gym.world_model.mdnrnn_gym import create_embed_rl_dataset, mdnrnn_gym
from ml.rl.thrift.core.ttypes import RLParameters
from ml.rl.training.rl_dataset import RLDataset
from ml.rl.training.world_model.mdnrnn_trainer import MDNRNNTrainer


logger = logging.getLogger(__name__)


def eval_mode(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        self = args[0]
        old_mdnrnn_mode = self.mdnrnn_trainer.mdnrnn.mdnrnn.training
        self.mdnrnn_trainer.mdnrnn.mdnrnn.eval()
        try:
            result = f(*args, **kwargs)
        finally:
            self.mdnrnn_trainer.mdnrnn.mdnrnn.train(old_mdnrnn_mode)
        return result

    return wrapper


class StateEmbedGymEnvironment(OpenAIGymEnvironment):
    def __init__(
        self,
        gymenv,
        epsilon,
        softmax_policy,
        gamma,
        epsilon_decay,
        minimum_epsilon,
        replay_buffer: OpenAIGymMemoryPool,
        mdnrnn_trainer: MDNRNNTrainer,
        max_embed_seq_len: int,
    ):
        self.max_embed_seq_len = max_embed_seq_len
        OpenAIGymEnvironment.__init__(
            self, gymenv, epsilon, softmax_policy, gamma, epsilon_decay, minimum_epsilon
        )
        # env normalization needs to be updated because stored states may have
        # been preprocessed
        state_mem = np.array([m[0] for m in replay_buffer.replay_memory])
        self.state_dim = state_mem.shape[1]
        self.state_min_value = np.amin(state_mem, axis=0)
        self.state_max_value = np.amax(state_mem, axis=0)
        self.mdnrnn_trainer = mdnrnn_trainer
        self.embed_size = self.mdnrnn_trainer.mdnrnn.num_hiddens
        assert not self.img, "Image-based environments not supported yet"
        self.recent_states = deque([], maxlen=self.max_embed_seq_len)
        self.recent_actions = deque([], maxlen=self.max_embed_seq_len)

    @property
    def normalization(self):
        if self.img:
            return None
        else:
            return only_continuous_normalizer(
                list(range(self.state_dim)), self.state_min_value, self.state_max_value
            )

    @eval_mode
    def transform_state(self, state):
        """ Embed state after either reset() or step() """
        assert len(self.recent_states) == len(self.recent_actions)

        if len(self.recent_states) == 0 and len(self.recent_actions) == 0:
            state_embed = np.hstack((np.zeros(self.embed_size), state))
            return state_embed

        # Embed the state as the hidden layer's output
        # until the previous step + current state
        mdnrnn_state = torch.tensor(
            list(self.recent_states), dtype=torch.float
        ).unsqueeze(1)
        mdnrnn_action = torch.tensor(
            list(self.recent_actions), dtype=torch.float
        ).unsqueeze(1)
        mdnrnn_input = rlt.StateAction(
            state=rlt.FeatureVector(float_features=mdnrnn_state),
            action=rlt.FeatureVector(float_features=mdnrnn_action),
        )
        mdnrnn_output = self.mdnrnn_trainer.mdnrnn(mdnrnn_input)
        hidden_embed = mdnrnn_output.all_steps_lstm_hidden[-1].squeeze()
        state_embed = np.hstack((hidden_embed.detach().numpy(), state))
        return state_embed

    def get_cur_state(self):
        return self.cur_state

    def reset(self):
        new_state = self.env.reset()
        assert len(new_state) == self.state_dim - self.embed_size
        self.recent_states = deque([], maxlen=self.max_embed_seq_len)
        self.recent_actions = deque([], maxlen=self.max_embed_seq_len)
        self.cur_state = new_state
        return new_state

    def step(self, action):
        if self.action_type == EnvType.DISCRETE_ACTION:
            action_np = np.zeros([self.action_dim], dtype=np.float32)
            action_np[action] = 1
        else:
            action_np = action
        self.recent_states.append(self.get_cur_state())
        self.recent_actions.append(action_np)
        res = self.env.step(action)
        next_state = res[0]
        assert len(next_state) == self.state_dim - self.embed_size
        self.cur_state = next_state
        return res

    def run_episode(
        self,
        predictor,
        max_steps=None,
        test=True,
        render=False,
        state_preprocessor=None,
    ):
        # StateEmbedGymEnvironment is only used for evaluation
        assert test and (not render) and (state_preprocessor is None)
        return super().run_episode(
            predictor, max_steps, test, render, state_preprocessor
        )


def main(args):
    parser = argparse.ArgumentParser(
        description="Train a RL net to play in an OpenAI Gym environment. "
        "States are embedded by a mdn-rnn model."
    )
    parser.add_argument(
        "-p",
        "--mdnrnn_parameters",
        help="Path to JSON parameters file for MDN-RNN training.",
    )
    parser.add_argument(
        "-q", "--rl_parameters", help="Path to JSON parameters file for RL training."
    )
    parser.add_argument(
        "-s",
        "--score-bar",
        help="Bar for averaged tests scores.",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        help="If set, will use GPU with specified ID. Otherwise will use CPU.",
        default=USE_CPU,
    )
    args = parser.parse_args(args)
    with open(args.mdnrnn_parameters, "r") as f:
        mdnrnn_params = json.load(f)
    with open(args.rl_parameters, "r") as f:
        rl_params = json.load(f)

    use_gpu = args.gpu_id != USE_CPU

    mdnrnn_trainer, embed_rl_dataset = create_mdnrnn_trainer_and_embed_dataset(
        mdnrnn_params, use_gpu
    )

    max_embed_seq_len = mdnrnn_params["run_details"]["seq_len"]
    _, _, rl_trainer, rl_predictor, state_embed_env = run_gym(
        rl_params,
        use_gpu,
        args.score_bar,
        embed_rl_dataset,
        mdnrnn_trainer,
        max_embed_seq_len,
    )


def create_mdnrnn_trainer_and_embed_dataset(mdnrnn_params, use_gpu):
    env, mdnrnn_trainer, _, _, _ = mdnrnn_gym(mdnrnn_params, use_gpu)
    embed_rl_dataset = RLDataset("/tmp/rl.pkl")
    create_embed_rl_dataset(
        env, mdnrnn_trainer, embed_rl_dataset, use_gpu, **mdnrnn_params["run_details"]
    )
    return mdnrnn_trainer, embed_rl_dataset


def run_gym(
    params,
    use_gpu,
    score_bar,
    embed_rl_dataset: RLDataset,
    mdnrnn_trainer: MDNRNNTrainer,
    max_embed_seq_len: int,
):
    rl_parameters = RLParameters(**params["rl"])
    env_type = params["env"]
    model_type = params["model_type"]
    epsilon, epsilon_decay, minimum_epsilon = create_epsilon(
        offline_train=True, rl_parameters=rl_parameters, params=params
    )

    replay_buffer = OpenAIGymMemoryPool(params["max_replay_memory_size"])
    for row in embed_rl_dataset.rows:
        replay_buffer.insert_into_memory(**row)

    state_embed_env = StateEmbedGymEnvironment(
        env_type,
        epsilon,
        rl_parameters.softmax_policy,
        rl_parameters.gamma,
        epsilon_decay,
        minimum_epsilon,
        replay_buffer,
        mdnrnn_trainer,
        max_embed_seq_len,
    )
    rl_trainer = create_trainer(
        params["model_type"], params, rl_parameters, use_gpu, state_embed_env
    )
    rl_predictor = create_predictor(
        rl_trainer, model_type, use_gpu, state_embed_env.action_dim
    )

    return train_gym_offline_rl(
        state_embed_env,
        replay_buffer,
        model_type,
        rl_trainer,
        rl_predictor,
        "{} offline rl state embed".format(env_type),
        score_bar,
        max_steps=params["run_details"]["max_steps"],
        avg_over_num_episodes=params["run_details"]["avg_over_num_episodes"],
        offline_train_epochs=params["run_details"]["offline_train_epochs"],
        bcq_imitator_hyper_params=None,
    )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = sys.argv
    main(args[1:])
