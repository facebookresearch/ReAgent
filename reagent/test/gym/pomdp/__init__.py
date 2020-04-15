#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

from reagent.test.environment import register_if_not_exists
from reagent.test.gym.pomdp.pocman import PocManEnv  # noqa
from reagent.test.gym.pomdp.string_game import StringGameEnv  # noqa


logger = logging.getLogger(__name__)


register_if_not_exists(
    id="Pocman-v0", entry_point="reagent.test.gym.pomdp.pocman:PocManEnv"
)
register_if_not_exists(
    id="StringGame-v0", entry_point="reagent.test.gym.pomdp.string_game:StringGameEnv"
)
