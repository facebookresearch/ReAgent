#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import itertools
import logging
import time
import unittest

import numpy as np
from ml.rl.test.gym.pomdp.pocman import ACTION_DICT, PocManEnv, random_action


logger = logging.getLogger(__name__)


class TestPOMDPEnvironment(unittest.TestCase):
    def setUp(self):
        logging.getLogger().setLevel(logging.DEBUG)

    def test_pocman(self):
        env = PocManEnv()
        env.seed(313)
        acc_rws = []

        for e in range(200):
            start_time = time.time()
            env.reset()
            acc_rw = 0
            for i in itertools.count():
                env.print_internal_state()
                action = random_action()
                ob, rw, done, info = env.step(action)
                print(
                    "After action {}: reward {}, observation {} ({})".format(
                        ACTION_DICT[action], rw, ob, env.print_ob(ob)
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
                print("")
            acc_rws.append(acc_rw)

        mean_acc_rw = np.mean(acc_rws)
        logger.debug("Average accumulated reward {}".format(mean_acc_rw))
        assert -80 <= mean_acc_rw <= -70
