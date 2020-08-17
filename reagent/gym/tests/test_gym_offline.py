#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
import os
import pprint
import unittest

import numpy as np

# pyre-fixme[21]: Could not find module `pytest`.
import pytest
import torch
from parameterized import parameterized
from reagent.core.types import RewardOptions
from reagent.gym.agents.agent import Agent
from reagent.gym.envs.gym import Gym
from reagent.gym.preprocessors import make_replay_buffer_trainer_preprocessor
from reagent.gym.runners.gymrunner import evaluate_for_n_episodes
from reagent.gym.utils import build_normalizer, fill_replay_buffer
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.runners.oss_batch_runner import OssBatchRunner
from reagent.tensorboardX import summary_writer_context
from reagent.test.base.horizon_test_base import HorizonTestBase
from reagent.workflow.model_managers.union import ModelManager__Union
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


try:
    # Use internal runner or OSS otherwise
    from reagent.runners.fb.fb_batch_runner import FbBatchRunner as BatchRunner
except ImportError:
    from reagent.runners.oss_batch_runner import OssBatchRunner as BatchRunner


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
    # pyre-fixme[56]: Argument `not torch.cuda.is_available()` to decorator factory
    #  `unittest.skipIf` could not be resolved in a global scope.
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gym_offline_gpu(self, name: str, config_path: str):
        self.run_from_config(
            run_test=run_test_offline,
            config_path=os.path.join(curr_dir, config_path),
            use_gpu=True,
        )
        logger.info(f"{name} passes!")


def evaluate_cem(env, manager, num_eval_episodes: int):
    # NOTE: for CEM, serving isn't implemented
    policy = manager.create_policy()
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
    use_gpu: bool,
):
    env = Gym(env_name=env_name)
    env.seed(SEED)
    env.action_space.seed(SEED)
    normalization = build_normalizer(env)
    logger.info(f"Normalization is: \n{pprint.pformat(normalization)}")

    manager = model.value
    runner = OssBatchRunner(
        use_gpu,
        manager,
        reward_options=RewardOptions(),
        normalization_data_map=normalization,
    )
    trainer = runner.initialize_trainer()

    # first fill the replay buffer to burn_in
    replay_buffer = ReplayBuffer(
        replay_capacity=replay_memory_size, batch_size=trainer.minibatch_size
    )
    # always fill full RB
    fill_replay_buffer(
        env=env, replay_buffer=replay_buffer, desired_size=replay_memory_size
    )

    device = torch.device("cuda") if use_gpu else None
    # pyre-fixme[6]: Expected `device` for 2nd param but got `Optional[torch.device]`.
    trainer_preprocessor = make_replay_buffer_trainer_preprocessor(trainer, device, env)

    writer = SummaryWriter()
    with summary_writer_context(writer):
        for epoch in range(num_train_epochs):
            logger.info(f"Evaluating before epoch {epoch}: ")
            eval_rewards = evaluate_cem(env, manager, 1)
            for _ in tqdm(range(num_batches_per_epoch)):
                train_batch = replay_buffer.sample_transition_batch()
                preprocessed_batch = trainer_preprocessor(train_batch)
                trainer.train(preprocessed_batch)

    logger.info(f"Evaluating after training for {num_train_epochs} epochs: ")
    eval_rewards = evaluate_cem(env, manager, num_eval_episodes)
    mean_rewards = np.mean(eval_rewards)
    assert (
        mean_rewards >= passing_score_bar
    ), f"{mean_rewards} doesn't pass the bar {passing_score_bar}."


if __name__ == "__main__":
    unittest.main()
