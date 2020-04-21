#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import gym
from gym_minigrid.wrappers import ReseedWrapper
from reagent.gym.envs.simple_minigrid import SimpleObsWrapper


class EnvFactory:
    @staticmethod
    def make(name: str) -> gym.Env:
        env = gym.make(name)
        env = ReseedWrapper(env)
        if name.startswith("MiniGrid-"):
            # Wrap in minigrid simplifier
            env = SimpleObsWrapper(env)
        return env
