#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

from gym.envs.registration import register
from ml.rl.test.gym.pomdp.pocman import PocManEnv  # noqa
from ml.rl.test.gym.pomdp.string_game import StringGameEnv  # noqa


logger = logging.getLogger(__name__)

register(id="Pocman-v0", entry_point="ml.rl.test.gym.pomdp.pocman:PocManEnv")
register(
    id="StringGame-v0", entry_point="ml.rl.test.gym.pomdp.string_game:StringGameEnv"
)
