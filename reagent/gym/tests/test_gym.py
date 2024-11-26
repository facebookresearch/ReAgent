#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
import logging
import os
import pprint
import unittest
import uuid
from typing import Optional

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from parameterized import parameterized
from reagent.gym.agents.agent import Agent
from reagent.gym.datasets.episodic_dataset import EpisodicDataset
from reagent.gym.datasets.replay_buffer_dataset import ReplayBufferDataset
from reagent.gym.envs import Env__Union
from reagent.gym.envs.env_wrapper import EnvWrapper
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.random_policies import make_random_policy_for_env
from reagent.gym.runners.gymrunner import evaluate_for_n_episodes
from reagent.gym.utils import build_normalizer, fill_replay_buffer
from reagent.model_managers.union import ModelManager__Union
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer

from reagent.test.base.horizon_test_base import HorizonTestBase


# for seeding the environment
SEED = 0
# exponential moving average parameter for tracking reward progress
REWARD_DECAY = 0.8
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


"""
Put on-policy gym tests here in the format (test name, path to yaml config).
Format path to be: "configs/<env_name>/<model_name>_<env_name>_online.yaml."
NOTE: These tests should ideally finish quickly (within 10 minutes) since they are
unit tests which are run many times.
"""
REPLAY_BUFFER_GYM_TESTS_1 = [
    ("Discrete CRR Cartpole", "configs/cartpole/discrete_crr_cartpole_online.yaml"),
    ("Discrete DQN Cartpole", "configs/cartpole/discrete_dqn_cartpole_online.yaml"),
    ("Discrete C51 Cartpole", "configs/cartpole/discrete_c51_cartpole_online.yaml"),
    ("Discrete QR Cartpole", "configs/cartpole/discrete_qr_cartpole_online.yaml"),
    (
        "Discrete DQN Open Gridworld",
        "configs/open_gridworld/discrete_dqn_open_gridworld.yaml",
    ),
    ("SAC Pendulum", "configs/pendulum/sac_pendulum_online.yaml"),
    ("Continuous CRR Pendulum", "configs/pendulum/continuous_crr_pendulum_online.yaml"),
    ("TD3 Pendulum", "configs/pendulum/td3_pendulum_online.yaml"),
]
REPLAY_BUFFER_GYM_TESTS_2 = [
    ("Parametric DQN Cartpole", "configs/cartpole/parametric_dqn_cartpole_online.yaml"),
    (
        "Parametric SARSA Cartpole",
        "configs/cartpole/parametric_sarsa_cartpole_online.yaml",
    ),
    # Disabled for now because flaky.
    # (
    #     "Sparse DQN Changing Arms",
    #     "configs/sparse/discrete_dqn_changing_arms_online.yaml",
    # ),
    ("SlateQ RecSim", "configs/recsim/slate_q_recsim_online.yaml"),
    (
        "SlateQ RecSim with Discount Scaled by Time Diff",
        "configs/recsim/slate_q_recsim_online_with_time_scale.yaml",
    ),
    (
        "SlateQ RecSim multi selection",
        "configs/recsim/slate_q_recsim_online_multi_selection.yaml",
    ),
    (
        "SlateQ RecSim multi selection average by current slate size",
        "configs/recsim/slate_q_recsim_online_multi_selection_avg_curr.yaml",
    ),
    ("PossibleActionsMask DQN", "configs/functionality/dqn_possible_actions_mask.yaml"),
]

ONLINE_EPISODE_GYM_TESTS = [
    (
        "REINFORCE Cartpole online",
        "configs/cartpole/discrete_reinforce_cartpole_online.yaml",
    ),
    (
        "PPO Cartpole online",
        "configs/cartpole/discrete_ppo_cartpole_online.yaml",
    ),
]


curr_dir = os.path.dirname(__file__)


class TestGym(HorizonTestBase):
    @parameterized.expand(REPLAY_BUFFER_GYM_TESTS_1)
    def test_replay_buffer_gym_cpu_1(self, name: str, config_path: str):
        self._test_replay_buffer_gym_cpu(name, config_path)

    @parameterized.expand(REPLAY_BUFFER_GYM_TESTS_2)
    def test_replay_buffer_gym_cpu_2(self, name: str, config_path: str):
        self._test_replay_buffer_gym_cpu(name, config_path)

    def _test_replay_buffer_gym_cpu(self, name: str, config_path: str):
        logger.info(f"Starting {name} on CPU")
        self.run_from_config(
            run_test=run_test_replay_buffer,
            config_path=os.path.join(curr_dir, config_path),
            use_gpu=False,
        )
        logger.info(f"{name} passes!")

    @parameterized.expand(REPLAY_BUFFER_GYM_TESTS_1)
    @pytest.mark.serial
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_replay_buffer_gym_gpu_1(self, name: str, config_path: str):
        self._test_replay_buffer_gym_gpu(name, config_path)

    @parameterized.expand(REPLAY_BUFFER_GYM_TESTS_2)
    @pytest.mark.serial
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_replay_buffer_gym_gpu_2(self, name: str, config_path: str):
        self._test_replay_buffer_gym_gpu(name, config_path)

    def _test_replay_buffer_gym_gpu(self, name: str, config_path: str):
        logger.info(f"Starting {name} on GPU")
        self.run_from_config(
            run_test=run_test_replay_buffer,
            config_path=os.path.join(curr_dir, config_path),
            use_gpu=True,
        )
        logger.info(f"{name} passes!")

    @parameterized.expand(ONLINE_EPISODE_GYM_TESTS)
    def test_online_episode_gym_cpu(self, name: str, config_path: str):
        logger.info(f"Starting {name} on CPU")
        self.run_from_config(
            run_test=run_test_online_episode,
            config_path=os.path.join(curr_dir, config_path),
            use_gpu=False,
        )
        logger.info(f"{name} passes!")


def eval_policy(
    env: EnvWrapper,
    serving_policy: Policy,
    num_eval_episodes: int,
    serving: bool = True,
) -> np.ndarray:
    agent = (
        Agent.create_for_env_with_serving_policy(env, serving_policy)
        if serving
        else Agent.create_for_env(env, serving_policy)
    )

    eval_rewards = evaluate_for_n_episodes(
        n=num_eval_episodes,
        env=env,
        agent=agent,
        max_steps=env.max_steps,
        num_processes=1,
    ).squeeze(1)

    logger.info("============Eval rewards==============")
    logger.info(eval_rewards)
    mean_eval = np.mean(eval_rewards)
    logger.info(f"average: {mean_eval};\tmax: {np.max(eval_rewards)}")
    return np.array(eval_rewards)


def identity_collate(batch):
    assert isinstance(batch, list) and len(batch) == 1, f"Got {batch}"
    return batch[0]


def run_test_replay_buffer(
    env: Env__Union,
    model: ModelManager__Union,
    replay_memory_size: int,
    train_every_ts: int,
    train_after_ts: int,
    num_train_episodes: int,
    passing_score_bar: float,
    num_eval_episodes: int,
    use_gpu: bool,
    minibatch_size: Optional[int] = None,
):
    """
    Run an online learning test with a replay buffer. The replay buffer is pre-filled, then the training starts.
    Each transition is added to the replay buffer immediately after it takes place.
    """
    env = env.value
    pl.seed_everything(SEED)
    env.seed(SEED)
    env.action_space.seed(SEED)

    normalization = build_normalizer(env)
    logger.info(f"Normalization is: \n{pprint.pformat(normalization)}")

    manager = model.value
    trainer = manager.build_trainer(
        use_gpu=use_gpu,
        normalization_data_map=normalization,
    )
    training_policy = manager.create_policy(trainer, serving=False)

    if not isinstance(trainer, pl.LightningModule):
        if minibatch_size is None:
            minibatch_size = trainer.minibatch_size
        assert minibatch_size == trainer.minibatch_size

    assert minibatch_size is not None

    replay_buffer = ReplayBuffer(
        replay_capacity=replay_memory_size, batch_size=minibatch_size
    )

    device = torch.device("cuda") if use_gpu else torch.device("cpu")
    # first fill the replay buffer using random policy
    train_after_ts = max(train_after_ts, minibatch_size)
    random_policy = make_random_policy_for_env(env)
    agent = Agent.create_for_env(env, policy=random_policy)
    fill_replay_buffer(
        env=env,
        replay_buffer=replay_buffer,
        desired_size=train_after_ts,
        agent=agent,
    )

    agent = Agent.create_for_env(env, policy=training_policy, device=device)
    # TODO: Simplify this setup by creating LightningDataModule
    dataset = ReplayBufferDataset.create_for_trainer(
        trainer,
        env,
        agent,
        replay_buffer,
        batch_size=minibatch_size,
        training_frequency=train_every_ts,
        num_episodes=num_train_episodes,
        max_steps=200,
        device=device,
    )
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=identity_collate)
    torch.use_deterministic_algorithms(True, warn_only=True)
    pl_trainer = pl.Trainer(
        max_epochs=1,
        gpus=int(use_gpu),
        default_root_dir=f"lightning_log_{str(uuid.uuid4())}",
    )
    # Note: the fit() function below also evaluates the agent along the way
    # and adds the new transitions to the replay buffer, so it is training
    # on incrementally larger and larger buffers.
    pl_trainer.fit(trainer, data_loader)

    # TODO: Also check train_reward

    serving_policy = manager.create_policy(
        trainer, serving=True, normalization_data_map=normalization
    )

    eval_rewards = eval_policy(env, serving_policy, num_eval_episodes, serving=True)
    assert (
        eval_rewards.mean() >= passing_score_bar
    ), f"Eval reward is {eval_rewards.mean()}, less than < {passing_score_bar}.\n"


def run_test_online_episode(
    env: Env__Union,
    model: ModelManager__Union,
    num_train_episodes: int,
    passing_score_bar: float,
    num_eval_episodes: int,
    use_gpu: bool,
):
    """
    Run an online learning test. At the end of each episode training is run on the trajectory.
    """
    env = env.value
    pl.seed_everything(SEED)
    env.seed(SEED)
    env.action_space.seed(SEED)

    normalization = build_normalizer(env)
    logger.info(f"Normalization is: \n{pprint.pformat(normalization)}")

    manager = model.value
    trainer = manager.build_trainer(
        use_gpu=use_gpu,
        normalization_data_map=normalization,
    )
    policy = manager.create_policy(trainer, serving=False)

    device = torch.device("cuda") if use_gpu else torch.device("cpu")

    agent = Agent.create_for_env(env, policy, device=device)

    torch.use_deterministic_algorithms(True, warn_only=True)
    pl_trainer = pl.Trainer(
        max_epochs=1,
        gpus=int(use_gpu),
        default_root_dir=f"lightning_log_{str(uuid.uuid4())}",
    )
    dataset = EpisodicDataset(
        env=env, agent=agent, num_episodes=num_train_episodes, seed=SEED
    )
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=identity_collate)
    pl_trainer.fit(trainer, data_loader)

    eval_rewards = evaluate_for_n_episodes(
        n=num_eval_episodes,
        env=env,
        agent=agent,
        max_steps=env.max_steps,
        num_processes=1,
    ).squeeze(1)
    assert (
        eval_rewards.mean() >= passing_score_bar
    ), f"Eval reward is {eval_rewards.mean()}, less than < {passing_score_bar}.\n"


if __name__ == "__main__":
    unittest.main()
