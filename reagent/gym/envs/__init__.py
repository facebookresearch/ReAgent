#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from reagent.workflow import types

from .changing_arms import ChangingArms  # noqa
from .dynamics.linear_dynamics import LinDynaEnv  # noqa
from .env_wrapper import EnvWrapper
from .gym import Gym  # noqa
from .pomdp.pocman import PocManEnv  # noqa
from .pomdp.string_game import StringGameEnv  # noqa
from .utils import register_if_not_exists


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


######## Register EnvWrappers ##########


try:
    from .recsim import RecSim  # noqa

    HAS_RECSIM = True
except ImportError:
    HAS_RECSIM = False

__all__ = list(
    filter(
        None, ["Env__Union", "Gym", "ChangingArms", "RecSim" if HAS_RECSIM else None]
    )
)


@EnvWrapper.fill_union()
class Env__Union(types.TaggedUnion):
    pass
