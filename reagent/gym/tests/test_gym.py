#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
Environments that require short training and evaluation time (<=10min)
can be tested in this file.
"""
import json
import logging
import random
import unittest
from typing import Optional

import gym
import numpy as np
import torch
from reagent.gym.agents.agent import Agent
from reagent.gym.agents.replay_buffer_add_fn import replay_buffer_add_fn
from reagent.gym.agents.replay_buffer_train_fn import replay_buffer_train_fn
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.continuous_sampler import GaussianSampler
from reagent.gym.policies.samplers.discrete_sampler import SoftmaxActionSampler
from reagent.gym.policies.scorers.continuous_scorer import sac_scorer
from reagent.gym.policies.scorers.discrete_scorer import (
    discrete_dqn_scorer,
    parametric_dqn_scorer,
)
from reagent.gym.preprocessors.action_preprocessors.action_preprocessor import (
    continuous_action_preprocessor,
    discrete_action_preprocessor,
)
from reagent.gym.preprocessors.policy_preprocessors.policy_preprocessor import (
    numpy_policy_preprocessor,
    tiled_numpy_policy_preprocessor,
)
from reagent.gym.preprocessors.trainer_preprocessors.trainer_preprocessor import (
    discrete_dqn_trainer_preprocessor,
    parametric_dqn_trainer_preprocessor,
    sac_trainer_preprocessor,
)
from reagent.gym.runners.gymrunner import run_episode
from reagent.json_serialize import from_json
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer, ReplayElement
from reagent.tensorboardX import SummaryWriterContext
from reagent.test.gym.open_ai_gym_environment import OpenAIGymEnvironment
from reagent.test.gym.run_gym import OpenAiGymParameters, create_trainer


DISCRETE_DQN_CARTPOLE_JSON = "ml/rl/test/gym/discrete_dqn_cartpole_v0.json"
DISCRETE_DQN_CARTPOLE_NUM_EPISODES = 50
PARAMETRIC_DQN_CARTPOLE_JSON = "ml/rl/test/gym/parametric_dqn_cartpole_v0.json"
PARAMETRIC_DQN_CARTPOLE_NUM_EPISODES = 50
# Though maximal score is 200, we set a lower bar to let tests finish in time
CARTPOLE_SCORE_BAR = 100

SAC_PENDULUM_JSON = "ml/rl/test/gym/sac_pendulum_v0.json"
SAC_PENDULUM_NUM_EPISODES = 50
# Though maximal score is 0, we set lower bar to let tests finish in time
PENDULUM_SCORE_BAR = -750

SEED = 0


def extract_config(config_path: str) -> OpenAiGymParameters:
    with open(config_path, "r") as f:
        json_data = json.loads(f.read())
        json_data["evaluation"] = {
            "calc_cpe_in_training": False
        }  # Slow without disabling
        json_data["use_gpu"] = False

    return from_json(json_data, OpenAiGymParameters)


def build_trainer(config):
    return create_trainer(config, OpenAIGymEnvironment(config.env))


def run(env: gym.Env, agent: Agent, num_episodes: int, max_steps: Optional[int] = None):
    reward_history = []
    for i in range(num_episodes):
        print(f"running episode {i}")
        ep_reward = run_episode(env, agent, max_steps)
        reward_history.append(ep_reward)
    return reward_history


def run_discrete_dqn_cartpole(config):
    trainer = build_trainer(config)
    num_episodes = DISCRETE_DQN_CARTPOLE_NUM_EPISODES
    env = gym.make(config.env)
    wrapped_env = OpenAIGymEnvironment(config.env)
    action_shape = np.array(wrapped_env.actions).shape
    action_type = np.int32
    replay_buffer = ReplayBuffer(
        observation_shape=env.reset().shape,
        stack_size=1,
        replay_capacity=config.max_replay_memory_size,
        batch_size=trainer.minibatch_size,
        observation_dtype=np.float32,
        action_shape=action_shape,
        action_dtype=action_type,
        reward_shape=(),
        reward_dtype=np.float32,
        extra_storage_types=[
            ReplayElement("possible_actions_mask", action_shape, action_type),
            ReplayElement("log_prob", (), np.float32),
        ],
    )

    actions = wrapped_env.actions
    normalization = wrapped_env.normalization
    policy = Policy(
        scorer=discrete_dqn_scorer(trainer.q_network),
        sampler=SoftmaxActionSampler(),
        policy_preprocessor=numpy_policy_preprocessor(),
    )
    agent = Agent(
        policy=policy,
        action_preprocessor=discrete_action_preprocessor,
        replay_buffer=replay_buffer,
        replay_buffer_add_fn=replay_buffer_add_fn,
        replay_buffer_train_fn=replay_buffer_train_fn(
            trainer=trainer,
            trainer_preprocessor=discrete_dqn_trainer_preprocessor(
                len(actions), normalization
            ),
            training_freq=config.run_details.train_every_ts,
            batch_size=trainer.minibatch_size,
            replay_burnin=config.run_details.train_after_ts,
        ),
    )

    reward_history = run(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        max_steps=config.run_details.max_steps,
    )
    return reward_history


def run_parametric_dqn_cartpole(config):
    trainer = build_trainer(config)
    num_episodes = PARAMETRIC_DQN_CARTPOLE_NUM_EPISODES
    env = gym.make(config.env)
    wrapped_env = OpenAIGymEnvironment(config.env)
    action_shape = np.array(wrapped_env.actions).shape
    action_type = np.float32
    replay_buffer = ReplayBuffer(
        observation_shape=env.reset().shape,
        stack_size=1,
        replay_capacity=config.max_replay_memory_size,
        batch_size=trainer.minibatch_size,
        observation_dtype=np.float32,
        action_shape=action_shape,
        action_dtype=action_type,
        reward_shape=(),
        reward_dtype=np.float32,
        extra_storage_types=[
            ReplayElement("possible_actions_mask", action_shape, action_type),
            ReplayElement("log_prob", (), np.float32),
        ],
    )

    actions = wrapped_env.actions
    normalization = wrapped_env.normalization

    policy = Policy(
        scorer=parametric_dqn_scorer(len(actions), trainer.q_network),
        sampler=SoftmaxActionSampler(),
        policy_preprocessor=tiled_numpy_policy_preprocessor(len(actions)),
    )
    agent = Agent(
        policy=policy,
        action_preprocessor=discrete_action_preprocessor,
        replay_buffer=replay_buffer,
        replay_buffer_add_fn=replay_buffer_add_fn,
        replay_buffer_train_fn=replay_buffer_train_fn(
            trainer=trainer,
            trainer_preprocessor=parametric_dqn_trainer_preprocessor(
                len(actions), normalization
            ),
            training_freq=config.run_details.train_every_ts,
            batch_size=trainer.minibatch_size,
            replay_burnin=config.run_details.train_after_ts,
        ),
    )

    reward_history = run(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        max_steps=config.run_details.max_steps,
    )
    return reward_history


def run_sac_pendulum(config):
    trainer = build_trainer(config)
    num_episodes = SAC_PENDULUM_NUM_EPISODES
    env = gym.make(config.env)
    action_shape = (1,)
    action_type = np.float32
    replay_buffer = ReplayBuffer(
        observation_shape=env.reset().shape,
        stack_size=1,
        replay_capacity=config.max_replay_memory_size,
        batch_size=trainer.minibatch_size,
        observation_dtype=np.float32,
        action_shape=action_shape,
        action_dtype=action_type,
        reward_shape=(),
        reward_dtype=np.float32,
        extra_storage_types=[
            ReplayElement("possible_actions_mask", action_shape, action_type),
            ReplayElement("log_prob", (), np.float32),
        ],
    )

    policy = Policy(
        scorer=sac_scorer(trainer.actor_network),
        sampler=GaussianSampler(
            trainer.actor_network,
            trainer.min_action_range_tensor_serving,
            trainer.max_action_range_tensor_serving,
            trainer.min_action_range_tensor_training,
            trainer.max_action_range_tensor_training,
        ),
        policy_preprocessor=numpy_policy_preprocessor(),
    )
    agent = Agent(
        policy=policy,
        action_preprocessor=continuous_action_preprocessor,
        replay_buffer=replay_buffer,
        replay_buffer_add_fn=replay_buffer_add_fn,
        replay_buffer_train_fn=replay_buffer_train_fn(
            trainer=trainer,
            trainer_preprocessor=sac_trainer_preprocessor(),
            training_freq=config.run_details.train_every_ts,
            batch_size=trainer.minibatch_size,
            replay_burnin=config.run_details.train_after_ts,
        ),
    )

    reward_history = run(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        max_steps=config.run_details.max_steps,
    )
    return reward_history


class TestGym(unittest.TestCase):
    def setUp(self):
        logging.getLogger().setLevel(logging.INFO)
        SummaryWriterContext._reset_globals()
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        random.seed(SEED)

    def test_discrete_dqn_cartpole(self):
        config = extract_config(DISCRETE_DQN_CARTPOLE_JSON)
        self.assertTrue(config.model_type == "pytorch_discrete_dqn")
        reward_history = run_discrete_dqn_cartpole(config)
        self.assertTrue(
            reward_history[-1] >= CARTPOLE_SCORE_BAR,
            "reward after %d episodes is %f < %f...\nFull reward history: %s"
            % (
                len(reward_history),
                reward_history[-1],
                CARTPOLE_SCORE_BAR,
                reward_history,
            ),
        )

    @unittest.skip("Skipping since training takes more than 10 min.")
    def test_parametric_dqn_cartpole(self):
        config = extract_config(PARAMETRIC_DQN_CARTPOLE_JSON)
        self.assertTrue(config.model_type == "pytorch_parametric_dqn")
        reward_history = run_parametric_dqn_cartpole(config)
        self.assertTrue(
            reward_history[-1] >= CARTPOLE_SCORE_BAR,
            "reward after %d episodes is %f < %f\nFull reward history: %s"
            % (
                len(reward_history),
                reward_history[-1],
                CARTPOLE_SCORE_BAR,
                reward_history,
            ),
        )

    @unittest.skip("Skipping since training takes more than 10 min.")
    def test_sac_pendulum(self):
        config = extract_config(SAC_PENDULUM_JSON)
        self.assertTrue(config.model_type == "soft_actor_critic")
        reward_history = run_sac_pendulum(config)
        self.assertTrue(
            reward_history[-1] >= PENDULUM_SCORE_BAR,
            "reward after %d episodes is %f < %f\nFull reward history: %s"(
                len(reward_history),
                reward_history[-1],
                PENDULUM_SCORE_BAR,
                reward_history,
            ),
        )
