#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
import os
import pprint
import unittest
import uuid

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from parameterized import parameterized
from reagent.fb.preprocessing.batch_preprocessor import identity_collate
from reagent.gym.agents.agent import Agent
from reagent.gym.datasets.replay_buffer_dataset import OfflineReplayBufferDataset
from reagent.gym.envs import Gym
from reagent.gym.policies.random_policies import make_random_policy_for_env
from reagent.gym.runners.gymrunner import evaluate_for_n_episodes
from reagent.gym.utils import build_normalizer, fill_replay_buffer
from reagent.model_managers.union import ModelManager__Union
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.test.base.horizon_test_base import HorizonTestBase
from reagent.workflow.types import RewardOptions

# for seeding the environment
SEED = 0
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


"""
These are trained offline.
"""
GYM_TESTS = [
    ("CEM Cartpole", "configs/world_model/cem_cartpole_offline.yaml"),
    (
        "CEM Single World Model Linear Dynamics",
        "configs/world_model/cem_single_world_model_linear_dynamics_offline.yaml",
    ),
    (
        "CEM Many World Models Linear Dynamics",
        "configs/world_model/cem_many_world_models_linear_dynamics_offline.yaml",
    ),
]


curr_dir = os.path.dirname(__file__)


class TestGymOffline(HorizonTestBase):
    # pyre-fixme[16]: Module `parameterized` has no attribute `expand`.
    @parameterized.expand(GYM_TESTS)
    def test_gym_offline_cpu(self, name: str, config_path: str):
        self.run_from_config(
            run_test=run_test_offline,
            config_path=os.path.join(curr_dir, config_path),
            use_gpu=False,
        )
        logger.info(f"{name} passes!")

    # pyre-fixme[16]: Module `parameterized` has no attribute `expand`.
    @parameterized.expand(GYM_TESTS)
    @pytest.mark.serial
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gym_offline_gpu(self, name: str, config_path: str):
        self.run_from_config(
            run_test=run_test_offline,
            config_path=os.path.join(curr_dir, config_path),
            use_gpu=True,
        )
        logger.info(f"{name} passes!")


def evaluate_cem(env, manager, trainer_module, num_eval_episodes: int):
    # NOTE: for CEM, serving isn't implemented
    policy = manager.create_policy(trainer_module, serving=False)
    agent = Agent.create_for_env(env, policy)
    return evaluate_for_n_episodes(
        n=num_eval_episodes, env=env, agent=agent, max_steps=env.max_steps
    )


def run_test_offline(
    env_name: str,
    model: ModelManager__Union,
    replay_memory_size: int,
    num_batches_per_epoch: int,
    num_train_epochs: int,
    passing_score_bar: float,
    num_eval_episodes: int,
    minibatch_size: int,
    use_gpu: bool,
):
    env = Gym(env_name=env_name)
    env.seed(SEED)
    env.action_space.seed(SEED)
    normalization = build_normalizer(env)
    logger.info(f"Normalization is: \n{pprint.pformat(normalization)}")

    manager = model.value
    trainer = manager.build_trainer(
        use_gpu=use_gpu,
        reward_options=RewardOptions(),
        normalization_data_map=normalization,
    )

    # first fill the replay buffer to burn_in
    replay_buffer = ReplayBuffer(
        replay_capacity=replay_memory_size, batch_size=minibatch_size
    )
    # always fill full RB
    random_policy = make_random_policy_for_env(env)
    agent = Agent.create_for_env(env, policy=random_policy)
    fill_replay_buffer(
        env=env,
        replay_buffer=replay_buffer,
        desired_size=replay_memory_size,
        agent=agent,
    )

    device = torch.device("cuda") if use_gpu else None
    dataset = OfflineReplayBufferDataset.create_for_trainer(
        trainer,
        env,
        replay_buffer,
        batch_size=minibatch_size,
        num_batches=num_batches_per_epoch,
        device=device,
    )
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=identity_collate)
    pl_trainer = pl.Trainer(
        max_epochs=num_train_epochs,
        gpus=int(use_gpu),
        deterministic=True,
        default_root_dir=f"lightning_log_{str(uuid.uuid4())}",
    )
    pl_trainer.fit(trainer, data_loader)

    logger.info(f"Evaluating after training for {num_train_epochs} epochs: ")
    eval_rewards = evaluate_cem(env, manager, trainer, num_eval_episodes)
    mean_rewards = np.mean(eval_rewards)
    assert (
        mean_rewards >= passing_score_bar
    ), f"{mean_rewards} doesn't pass the bar {passing_score_bar}."


if __name__ == "__main__":
    unittest.main()
