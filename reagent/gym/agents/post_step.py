#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe


import logging

import gym
from reagent.gym.preprocessors import make_replay_buffer_inserter
from reagent.gym.types import Transition
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer


logger = logging.getLogger(__name__)


def add_replay_buffer_post_step(
    replay_buffer: ReplayBuffer,
    env: gym.Env,
    replay_buffer_inserter=None,
):
    """
    Simply add transitions to replay_buffer.
    """

    if replay_buffer_inserter is None:
        replay_buffer_inserter = make_replay_buffer_inserter(env)

    def post_step(transition: Transition) -> None:
        replay_buffer_inserter(replay_buffer, transition)

    return post_step
