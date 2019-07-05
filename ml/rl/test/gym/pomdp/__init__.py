#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

from gym.envs.registration import register, registry
from ml.rl.test.gym.pomdp.pocman import PocManEnv  # noqa
from ml.rl.test.gym.pomdp.string_game import StringGameEnv  # noqa


logger = logging.getLogger(__name__)


def register_if_not_exists(id, entry_point):
    """
    Preventing tests from failing trying to re-register environments
    """
    if id not in registry.env_specs:
        register(id=id, entry_point=entry_point)


register_if_not_exists(
    id="Pocman-v0", entry_point="ml.rl.test.gym.pomdp.pocman:PocManEnv"
)
register_if_not_exists(
    id="StringGame-v0", entry_point="ml.rl.test.gym.pomdp.string_game:StringGameEnv"
)
