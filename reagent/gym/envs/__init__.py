#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from .dynamics.linear_dynamics import LinDynaEnv  # noqa
from .env_factory import EnvFactory
from .pomdp.pocman import PocManEnv  # noqa
from .pomdp.string_game import StringGameEnv  # noqa
from .utils import register_if_not_exists


__all__ = ["EnvFactory"]


######### Register classes below ##########

CUR_MODULE = "reagent.gym.envs"
ENV_CLASSES = [
    ("Pocman-v0", ".pomdp.pocman:PocManEnv"),
    ("StringGame-v0", ".pomdp.string_game:StringGameEnv"),
    ("LinearDynamics-v0", ".dynamics.linear_dynamics:LinDynaEnv"),
]

for env_name, rel_module_path in ENV_CLASSES:
    full_module_path = CUR_MODULE + rel_module_path
    register_if_not_exists(id=env_name, entry_point=full_module_path)
