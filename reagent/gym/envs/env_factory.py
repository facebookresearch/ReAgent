#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import gym
from gym_minigrid.wrappers import ReseedWrapper
from reagent.gym.envs.simple_minigrid import SimpleObsWrapper


logger = logging.getLogger(__name__)


class EnvFactory:
    @staticmethod
    def make(name: str) -> gym.Env:
        env: gym.Env = gym.make(name)
        if name.startswith("MiniGrid-"):
            # Wrap in minigrid simplifier
            env = SimpleObsWrapper(ReseedWrapper(env))

        logger.info(
            f"Env: {name}; observation_space: {env.observation_space}; "
            f"action_space: {env.action_space}"
        )

        return env
