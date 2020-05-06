#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import time
import unittest

import numpy as np
import scipy.linalg as linalg
from reagent.gym.envs.env_factory import EnvFactory


logger = logging.getLogger(__name__)


class TestLinearDynamicsEnvironment(unittest.TestCase):
    def setUp(self):
        logging.getLogger().setLevel(logging.DEBUG)

    def test_random_vs_lqr(self):
        """
        Test random actions vs. a LQR controller. LQR controller should perform
        much better than random actions in the linear dynamics environment.
        """
        env = EnvFactory.make("LinearDynamics-v0")
        num_test_episodes = 500

        def random_policy(env, state):
            return np.random.uniform(
                env.action_space.low, env.action_space.high, env.action_dim
            )

        def lqr_policy(env, state):
            # Four matrices that characterize the environment
            A, B, Q, R = env.A, env.B, env.Q, env.R
            # Solve discrete algebraic Riccati equation:
            M = linalg.solve_discrete_are(A, B, Q, R)
            K = np.dot(
                linalg.inv(np.dot(np.dot(B.T, M), B) + R), (np.dot(np.dot(B.T, M), A))
            )
            state = state.reshape((-1, 1))
            action = -K.dot(state).squeeze()
            return action

        mean_acc_rws_random = self.run_n_episodes(env, num_test_episodes, random_policy)
        mean_acc_rws_lqr = self.run_n_episodes(env, num_test_episodes, lqr_policy)
        logger.info(f"Mean acc. reward of random policy: {mean_acc_rws_random}")
        logger.info(f"Mean acc. reward of LQR policy: {mean_acc_rws_lqr}")
        assert mean_acc_rws_lqr > mean_acc_rws_random

    def run_n_episodes(self, env, num_episodes, policy):
        acc_rws = []
        for e in range(num_episodes):
            start_time = time.time()
            ob = env.reset()
            acc_rw = 0
            for i in range(env.max_steps):
                action = policy(env, ob)
                ob, rw, done, info = env.step(action)
                print(
                    "After action {}: reward {}, observation {}".format(action, rw, ob)
                )
                acc_rw += rw
                if done:
                    logger.debug(
                        "Epoch {} done in {} step {} seconds, acc. reward {}".format(
                            e, i, time.time() - start_time, acc_rw
                        )
                    )
                    break
                print("")
            acc_rws.append(acc_rw)

        mean_acc_rw = np.mean(acc_rws)
        return mean_acc_rw
