#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging
import time
import unittest

import numpy as np
from reagent.gym.envs import Gym


logger = logging.getLogger(__name__)


class TestPOMDPEnvironment(unittest.TestCase):
    def setUp(self):
        logging.getLogger().setLevel(logging.DEBUG)

    def test_string_game_v1(self):
        env = Gym(env_name="StringGame-v1")
        env.seed(313)
        mean_acc_reward = self._test_env(env)
        assert 1.0 >= mean_acc_reward

    def test_string_game(self):
        env = Gym(env_name="StringGame-v0")
        env.seed(313)
        mean_acc_reward = self._test_env(env)
        assert 0.1 >= mean_acc_reward

    def test_pocman(self):
        env = Gym(env_name="Pocman-v0")
        env.seed(313)
        mean_acc_reward = self._test_env(env)
        assert -80 <= mean_acc_reward <= -70

    def _test_env(self, env):
        acc_rws = []
        num_test_episodes = 200

        for e in range(num_test_episodes):
            start_time = time.time()
            env.reset()
            acc_rw = 0
            for i in range(1, env.max_steps + 1):
                env.print_internal_state()
                action = env.random_action()
                ob, rw, done, info = env.step(action)
                logger.debug(
                    "After action {}: reward {}, observation {} ({})".format(
                        env.print_action(action), rw, ob, env.print_ob(ob)
                    )
                )
                acc_rw += rw
                if done:
                    env.print_internal_state()
                    logger.debug(
                        "Epoch {} done in {} step {} seconds, accumulated reward {}\n\n".format(
                            e, i, time.time() - start_time, acc_rw
                        )
                    )
                    break
            acc_rws.append(acc_rw)

        mean_acc_rw = np.mean(acc_rws)
        logger.debug("Average accumulated reward {}".format(mean_acc_rw))
        return mean_acc_rw
