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
from reagent.gym.agents.post_step import (
    add_replay_buffer_post_step,
    train_with_replay_buffer_post_step,
)
from reagent.gym.envs.env_factory import EnvFactory
from reagent.gym.preprocessors import (
    make_default_serving_action_extractor,
    make_default_serving_obs_preprocessor,
    make_replay_buffer_trainer_preprocessor,
)
from reagent.gym.runners.gymrunner import run_episode
from reagent.gym.tests.utils import build_normalizer
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.test.base.horizon_test_base import HorizonTestBase
from reagent.workflow.model_managers.union import ModelManager__Union
from reagent.workflow.types import RewardOptions
from tqdm import tqdm
from reagent.gym.policies.random_policies import make_random_policy_for_env

logger = logging.getLogger(__name__)
curr_dir = os.path.dirname(__file__)

SEED = 0


def print_mdnrnn_losses(epoch, batch_num, losses):
    logger.info(
        f"Printing loss for Epoch {epoch}, Batch {batch_num};\n"
        f"loss={losses['loss']}, bce={losses['bce']}, "
        f"gmm={losses['gmm']}, mse={losses['mse']} \n"
    )


def test_mdnrnn(
    env: str,
    model: ModelManager__Union,
    num_train_episodes: int,
    num_test_episodes: int,
    seq_len: int,
    batch_size: int,
    num_train_epochs: int,
    use_gpu: bool,
):
    env = EnvFactory.make(env)
    env.seed(SEED)

    normalization = build_normalizer(env)
    manager = model.value
    trainer = manager.initialize_trainer(
        use_gpu=use_gpu,
        reward_options=RewardOptions(),
        normalization_data_map=normalization,
    )

    def create_rb(episodes):
        rb = ReplayBuffer.create_from_env(
            env=env,
            replay_memory_size=episodes * env._max_episode_steps,
            # batch_size=batch_size,
            update_horizon=seq_len,
        )
        # policy = manager.create_policy()
        random_policy = make_random_policy_for_env(env)
        post_step = add_replay_buffer_post_step(rb)
        agent = Agent.create_for_env(
            env, policy=random_policy, post_transition_callback=post_step
        )
        for _ in tqdm(range(episodes)):
            run_episode(env, agent)
        return rb

    trainer_preprocessor = make_replay_buffer_trainer_preprocessor(trainer, use_gpu)
    test_replay_buffer = create_rb(num_test_episodes)

    train_replay_buffer = create_rb(num_train_episodes)
    num_batch_per_epoch = train_replay_buffer.size // batch_size
    for epoch in range(num_train_epochs):
        for i in range(num_batch_per_epoch):
            batch = train_replay_buffer.sample_transition_batch_tensor(
                batch_size=batch_size
            )
            preprocessed_batch = trainer_preprocessor(batch)
            losses = trainer.train(preprocessed_batch)
            print_mdnrnn_losses(epoch, i, losses)
            logger.info(
                f"cum loss={np.mean(trainer.cum_loss)}, "
                f"cum bce={np.mean(trainer.cum_bce)}, "
                f"cum gmm={np.mean(trainer.cum_gmm)}, "
                f"cum mse={np.mean(trainer.cum_mse)}\n"
            )

        # validation
        trainer.mdnrnn.mdnrnn.eval()
        test_batch = test_replay_buffer.sample_transition_batch_tensor(
            batch_size=batch_size
        )
        preprocessed_test_batch = trainer_preprocessor(test_batch)
        valid_losses = trainer.get_loss(preprocessed_test_batch)
        print_mdnrnn_losses(epoch, "validation", valid_losses)
        trainer.mdnrnn.mdnrnn.train()

    # feature_importance = calculate_feature_importance(...)
    # feature_sensitivity = calculate_feature_sensitivity_by_actions(...)
    # return feature_importance, feature_sensitivity


class TestWorldModel(HorizonTestBase):
    def test_mdnrnn(self):
        """ Test MDNRNN feature importance and feature sensitivity. """
        config_path = "configs/world_model/cartpole_features.yaml"
        self.run_from_config(
            run_test=test_mdnrnn,
            config_path=os.path.join(curr_dir, config_path),
            use_gpu=False,
        )
        logger.info("MDNRNN feature test passes!")

    @unittest.skip("Not Implemented")
    def test_world_model(self):
        """ Train DQN on POMDP given features from world model. """
        raise NotImplementedError()


if __name__ == "__main__":
    unittest.main()
