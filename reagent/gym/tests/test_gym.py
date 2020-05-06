#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
import os
import pprint
import random
import unittest
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from parameterized import parameterized
from reagent.core.configuration import make_config_class
from reagent.gym.agents.agent import Agent
from reagent.gym.agents.post_step import train_with_replay_buffer_post_step
from reagent.gym.envs.env_factory import EnvFactory
from reagent.gym.runners.gymrunner import run_episode
from reagent.parameters import NormalizationData, NormalizationKey
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.tensorboardX import SummaryWriterContext
from reagent.test.base.utils import (
    only_continuous_action_normalizer,
    only_continuous_normalizer,
)
from reagent.workflow.model_managers.union import ModelManager__Union
from reagent.workflow.types import RewardOptions
from ruamel.yaml import YAML


logger = logging.getLogger(__name__)
curr_dir = os.path.dirname(__file__)

SEED = 0


def build_state_normalizer(env):
    if isinstance(env.observation_space, gym.spaces.Box):
        assert (
            len(env.observation_space.shape) == 1
        ), f"{env.observation_space.shape} has dim > 1, and is not supported."
        return NormalizationData(
            dense_normalization_parameters=only_continuous_normalizer(
                list(range(env.observation_space.shape[0])),
                env.observation_space.low,
                env.observation_space.high,
            )
        )
    elif isinstance(env.observation_space, gym.spaces.Dict):
        # assuming env.observation_space is image
        return None
    else:
        raise NotImplementedError(f"{env.observation_space} not supported")


def build_action_normalizer(env):
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Discrete):
        return only_continuous_normalizer(
            list(range(action_space.n)), min_value=0, max_value=1
        )
    elif isinstance(action_space, gym.spaces.Box):
        assert action_space.shape == (
            1,
        ), f"Box action shape {action_space.shape} not supported."

        return NormalizationData(
            dense_normalization_parameters=only_continuous_action_normalizer(
                [0],
                min_value=action_space.low.item(),
                max_value=action_space.high.item(),
            )
        )
    else:
        raise NotImplementedError(f"{action_space} not supported.")


def build_normalizer(env):
    return {
        NormalizationKey.STATE: build_state_normalizer(env),
        NormalizationKey.ACTION: build_action_normalizer(env),
    }


def run_test(
    env: str,
    model: ModelManager__Union,
    replay_memory_size: int,
    train_every_ts: int,
    train_after_ts: int,
    num_train_episodes: int,
    max_steps: Optional[int],
    passing_score_bar: float,
    num_eval_episodes: int,
    use_gpu: bool,
):
    env = EnvFactory.make(env)
    env.seed(SEED)
    env.action_space.seed(SEED)
    normalization = build_normalizer(env)
    logger.info(f"Normalization is: \n{pprint.pformat(normalization)}")

    manager = model.value
    trainer = manager.initialize_trainer(
        use_gpu=use_gpu,
        reward_options=RewardOptions(),
        normalization_data_map=normalization,
    )

    replay_buffer = ReplayBuffer.create_from_env(
        env=env,
        replay_memory_size=replay_memory_size,
        batch_size=trainer.minibatch_size,
    )

    device = torch.device("cuda") if use_gpu else None
    post_step = train_with_replay_buffer_post_step(
        replay_buffer=replay_buffer,
        trainer=trainer,
        training_freq=train_every_ts,
        batch_size=trainer.minibatch_size,
        replay_burnin=train_after_ts,
        device=device,
    )

    training_policy = manager.create_policy(serving=False)
    agent = Agent.create_for_env(
        env, policy=training_policy, post_transition_callback=post_step, device=device
    )

    train_rewards = []
    for i in range(num_train_episodes):
        ep_reward = run_episode(env=env, agent=agent, max_steps=max_steps)
        train_rewards.append(ep_reward)
        logger.info(f"Finished training episode {i} with reward {ep_reward}.")

    assert train_rewards[-1] >= passing_score_bar, (
        f"reward after {len(train_rewards)} episodes is {train_rewards[-1]},"
        f"less than < {passing_score_bar}...\n"
        f"Full reward history: {train_rewards}"
    )

    logger.info("============Train rewards=============")
    logger.info(train_rewards)

    serving_policy = manager.create_policy(serving=True)
    agent = Agent.create_from_serving_policy(serving_policy, env)

    eval_rewards = []
    for i in range(num_eval_episodes):
        ep_reward = run_episode(env=env, agent=agent, max_steps=max_steps)
        eval_rewards.append(ep_reward)
        logger.info(f"Finished eval episode {i} with reward {ep_reward}.")

    assert np.mean(eval_rewards) >= passing_score_bar, (
        f"Predictor reward is {np.mean(eval_rewards)},"
        f"less than < {passing_score_bar}...\n"
        f"Full eval rewards: {eval_rewards}."
    )

    logger.info("============Eval rewards==============")
    logger.info(eval_rewards)


def run_from_config(path, use_gpu):
    yaml = YAML(typ="safe")
    with open(path, "r") as f:
        config_dict = yaml.load(f.read())
    config_dict["use_gpu"] = use_gpu

    @make_config_class(run_test)
    class ConfigClass:
        pass

    config = ConfigClass(**config_dict)
    return run_test(**config.asdict())


GYM_TESTS = [
    ("Discrete Dqn Cartpole", "configs/cartpole/discrete_dqn_cartpole_online.yaml"),
    (
        "Discrete Dqn Open Gridworld",
        "configs/open_gridworld/discrete_dqn_open_gridworld.yaml",
    ),
    ("SAC Pendulum", "configs/pendulum/sac_pendulum_online.yaml"),
    ("TD3 Pendulum", "configs/pendulum/td3_pendulum_online.yaml"),
]


class TestGym(unittest.TestCase):
    """
    Environments that require short training time (<=10min) can be tested here.
    """

    def setUp(self):
        SummaryWriterContext._reset_globals()
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        random.seed(SEED)

    @parameterized.expand(GYM_TESTS)
    def test_gym_cpu(self, name: str, config_path: str):
        run_from_config(os.path.join(curr_dir, config_path), False)
        logger.info(f"{name} passes!")

    @parameterized.expand(GYM_TESTS)
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gym_gpu(self, name: str, config_path: str):
        run_from_config(os.path.join(curr_dir, config_path), True)
        logger.info(f"{name} passes!")


if __name__ == "__main__":
    unittest.main()
