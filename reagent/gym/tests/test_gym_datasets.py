#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

from reagent.gym.agents.agent import Agent
from reagent.gym.datasets.episodic_dataset import EpisodicDataset
from reagent.gym.envs import Gym
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.discrete_sampler import SoftmaxActionSampler
from reagent.gym.utils import build_normalizer
from reagent.net_builder.discrete_dqn.fully_connected import FullyConnected


logger = logging.getLogger(__name__)


class TestEpisodicDataset(unittest.TestCase):
    def setUp(self):
        logging.getLogger().setLevel(logging.DEBUG)
        env = Gym("CartPole-v0")
        norm = build_normalizer(env)
        net_builder = FullyConnected(sizes=[8], activations=["linear"])
        cartpole_scorer = net_builder.build_q_network(
            state_feature_config=None,
            state_normalization_data=norm["state"],
            output_dim=len(norm["action"].dense_normalization_parameters),
        )
        policy = Policy(scorer=cartpole_scorer, sampler=SoftmaxActionSampler())
        agent = Agent.create_for_env(env, policy)
        self.max_steps = 3
        self.num_episodes = 6
        self.dataset = EpisodicDataset(
            env=env,
            agent=agent,
            num_episodes=self.num_episodes,
            seed=0,
            max_steps=self.max_steps,
        )

    def test_episodic_dataset(self):
        pass
        num_batches = 0
        for batch in self.dataset:
            num_batches += 1
            self.assertLessEqual(len(batch["reward"]), self.max_steps)
            self.assertIsInstance(batch, dict)
        self.assertEqual(num_batches, self.num_episodes)
