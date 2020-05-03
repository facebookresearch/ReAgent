#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

from gym.envs.registration import register, registry


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def register_if_not_exists(id, entry_point):
    """
    Preventing tests from failing trying to re-register environments
    """
    if id not in registry.env_specs:
        logging.info(f"Registering id={id}, entry_point={entry_point}.")
        register(id=id, entry_point=entry_point)
