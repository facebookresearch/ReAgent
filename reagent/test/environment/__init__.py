#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging

from gym.envs.registration import register, registry
from reagent.test.environment.linear_dynamics import LinDynaEnv  # noqa


logger = logging.getLogger(__name__)


def register_if_not_exists(id, entry_point):
    """
    Preventing tests from failing trying to re-register environments
    """
    if id not in registry.env_specs:
        register(id=id, entry_point=entry_point)


register_if_not_exists(
    id="LinearDynamics-v0",
    entry_point="reagent.test.environment.linear_dynamics:LinDynaEnv",
)
