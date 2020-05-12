#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import gym
from reagent.gym.preprocessors import make_replay_buffer_inserter
from reagent.replay_memory import ReplayBuffer


class TestBasicReplayBufferInserter(unittest.TestCase):
    def test_cartpole(self):
        env = gym.make("CartPole-v0")
        replay_buffer = ReplayBuffer.create_from_env(
            env, replay_memory_size=10, batch_size=5
        )
        replay_buffer_inserter = make_replay_buffer_inserter(env)
        obs = env.reset()
        terminal = False
        i = 0
        while not terminal and i < 5:
            action = env.action_space.sample()
            next_obs, reward, terminal, _ = env.step(action)
            replay_buffer_inserter(
                replay_buffer, obs, action, reward, terminal, log_prob=0.0
            )
            obs = next_obs


class TestRecSimReplayBufferInserter(unittest.TestCase):
    pass
