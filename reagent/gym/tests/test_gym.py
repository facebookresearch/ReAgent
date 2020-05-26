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
from reagent.gym.runners.gymrunner import evaluate_for_n_episodes, run_episode
from reagent.gym.utils import build_normalizer, fill_replay_buffer
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.tensorboardX import summary_writer_context
from reagent.test.base.horizon_test_base import HorizonTestBase
from reagent.workflow.model_managers.union import ModelManager__Union
from reagent.workflow.types import RewardOptions
from torch.utils.tensorboard import SummaryWriter


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
    ("Discrete DQN Cartpole", "configs/cartpole/discrete_dqn_cartpole_online.yaml"),
    ("Discrete C51 Cartpole", "configs/cartpole/discrete_c51_cartpole_online.yaml"),
    ("Discrete QR Cartpole", "configs/cartpole/discrete_qr_cartpole_online.yaml"),
    (
        "Discrete DQN Open Gridworld",
        "configs/open_gridworld/discrete_dqn_open_gridworld.yaml",
    ),
    ("SAC Pendulum", "configs/pendulum/sac_pendulum_online.yaml"),
    ("TD3 Pendulum", "configs/pendulum/td3_pendulum_online.yaml"),
    ("Parametric DQN Cartpole", "configs/cartpole/parametric_dqn_cartpole_online.yaml"),
    (
        "Parametric SARSA Cartpole",
        "configs/cartpole/parametric_sarsa_cartpole_online.yaml",
    ),
]


curr_dir = os.path.dirname(__file__)


class TestGym(HorizonTestBase):
    # pyre-fixme[16]: Module `parameterized` has no attribute `expand`.
    @parameterized.expand(GYM_TESTS)
    def test_gym_cpu(self, name: str, config_path: str):
        self.run_from_config(
            run_test=run_test,
            config_path=os.path.join(curr_dir, config_path),
            use_gpu=False,
        )
        logger.info(f"{name} passes!")

    # pyre-fixme[16]: Module `parameterized` has no attribute `expand`.
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
    env_name: str,
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
    env = EnvFactory.make(env_name)
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
    training_policy = manager.create_policy(serving=False)

    replay_buffer = ReplayBuffer.create_from_env(
        env=env,
        replay_memory_size=replay_memory_size,
        batch_size=trainer.minibatch_size,
    )

    device = torch.device("cuda") if use_gpu else None
    # first fill the replay buffer to burn_in
    train_after_ts = max(train_after_ts, trainer.minibatch_size)
    fill_replay_buffer(
        env=env, replay_buffer=replay_buffer, desired_size=train_after_ts
    )

    post_step = train_with_replay_buffer_post_step(
        replay_buffer=replay_buffer,
        env=env,
        trainer=trainer,
        training_freq=train_every_ts,
        batch_size=trainer.minibatch_size,
        device=device,
    )

    agent = Agent.create_for_env(
        env,
        policy=training_policy,
        post_transition_callback=post_step,
        # pyre-fixme[6]: Expected `Union[str, torch.device]` for 4th param but got
        #  `Optional[torch.device]`.
        device=device,
    )

    writer = SummaryWriter()
    with summary_writer_context(writer):
        train_rewards = []
        for i in range(num_train_episodes):
            trajectory = run_episode(
                env=env, agent=agent, mdp_id=i, max_steps=max_steps
            )
            ep_reward = trajectory.calculate_cumulative_reward()
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
    agent = Agent.create_for_env_with_serving_policy(env, serving_policy)

    eval_rewards = evaluate_for_n_episodes(
        n=num_eval_episodes, env=env, agent=agent, max_steps=max_steps
    ).squeeze(1)
    assert np.mean(eval_rewards) >= passing_score_bar, (
        f"Predictor reward is {np.mean(eval_rewards)},"
        f"less than < {passing_score_bar}...\n"
        f"Full eval rewards: {eval_rewards}."
    )

    logger.info("============Eval rewards==============")
    logger.info(eval_rewards)


if __name__ == "__main__":
    unittest.main()
