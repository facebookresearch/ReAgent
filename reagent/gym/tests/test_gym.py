#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
import os
import pprint
import unittest
from typing import Optional

import numpy as np
import torch
from parameterized import parameterized
from reagent.gym.agents.agent import Agent
from reagent.gym.agents.post_step import train_with_replay_buffer_post_step
from reagent.gym.envs.env_factory import EnvFactory
from reagent.gym.runners.gymrunner import run_episode
from reagent.gym.tests.utils import build_normalizer
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.test.base.horizon_test_base import HorizonTestBase

from reagent.workflow.model_managers.union import ModelManager__Union
from reagent.workflow.types import RewardOptions


# for seeding the environment
SEED = 0
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


"""
Put on-policy gym tests here in the format (test name, path to yaml config).
Format path to be: "configs/<env_name>/<model_name>_<env_name>_online.yaml."
NOTE: These tests should ideally finish quickly (within 10 minutes) since they are
unit tests which are run many times.
"""
GYM_TESTS = [
    ("Discrete Dqn Cartpole", "configs/cartpole/discrete_dqn_cartpole_online.yaml"),
    (
        "Discrete Dqn Open Gridworld",
        "configs/open_gridworld/discrete_dqn_open_gridworld.yaml",
    ),
    ("SAC Pendulum", "configs/pendulum/sac_pendulum_online.yaml"),
    ("TD3 Pendulum", "configs/pendulum/td3_pendulum_online.yaml"),
]


curr_dir = os.path.dirname(__file__)


class TestGym(HorizonTestBase):
    @parameterized.expand(GYM_TESTS)
    def test_gym_cpu(self, name: str, config_path: str):
        self.run_from_config(
            run_test=run_test,
            config_path=os.path.join(curr_dir, config_path),
            use_gpu=False,
        )
        logger.info(f"{name} passes!")

    @parameterized.expand(GYM_TESTS)
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gym_gpu(self, name: str, config_path: str):
        self.run_from_config(
            run_test=run_test,
            config_path=os.path.join(curr_dir, config_path),
            use_gpu=True,
        )
        logger.info(f"{name} passes!")


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

    def gym_to_reagent_serving(obs: np.array) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = torch.tensor(obs).float().unsqueeze(0)
        presence_tensor = torch.ones_like(obs_tensor)
        return (obs_tensor, presence_tensor)

    serving_policy = manager.create_policy(serving=True)
    agent = Agent.create_for_env(
        env, policy=serving_policy, obs_preprocessor=gym_to_reagent_serving
    )

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


if __name__ == "__main__":
    unittest.main()
