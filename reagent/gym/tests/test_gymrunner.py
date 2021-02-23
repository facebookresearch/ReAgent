#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

# from reagent.gym.agents.agent import Agent
# from reagent.gym.envs import Gym
# from reagent.gym.policies.policy import Policy
# from reagent.gym.policies.samplers.discrete_sampler import SoftmaxActionSampler
# from reagent.gym.utils import build_normalizer
# from reagent.types import PolicyGradientInput
# from reagent.gym.runners.gymrunner import create_trajectory_dataloader
# from reagent.net_builder.discrete_dqn.fully_connected import FullyConnected


logger = logging.getLogger(__name__)


class TestGymrunner(unittest.TestCase):
    def setUp(self):
        logging.getLogger().setLevel(logging.DEBUG)
        # env = Gym("CartPole-v0")
        # norm = build_normalizer(env)
        # net_builder = FullyConnected(sizes=[8], activations=["linear"])
        # cartpole_scorer = net_builder.build_q_network(
        #     state_feature_config=None,
        #     state_normalization_data=norm["state"],
        #     output_dim=len(norm["action"].dense_normalization_parameters),
        # )
        # policy = Policy(scorer=cartpole_scorer, sampler=SoftmaxActionSampler())
        # agent = Agent.create_for_env(env, policy)
        # self.max_steps = 38
        # self.num_episodes = 29
        # self.dataloader = create_trajectory_dataloader(
        #     env=env,
        #     agent=agent,
        #     num_episodes=self.num_episodes,
        #     seed=0,
        #     max_steps=self.max_steps,
        # )

    def test_create_trajectory_dataloader(self):
        pass
        # num_batches = 0
        # for batch in self.dataloader:
        #     num_batches += 1
        #     self.assertLessEqual(len(batch), self.max_steps)
        #     self.assertIsInstance(batch, PolicyGradientInput)
        # self.assertEqual(num_batches, self.num_episodes)
