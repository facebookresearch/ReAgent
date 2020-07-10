#!/usr/bin/env python3

import logging
from typing import Optional

import gym
from gym_minigrid.wrappers import ReseedWrapper
from reagent.core.dataclasses import dataclass
from reagent.gym.envs.env_wrapper import EnvWrapper
from reagent.gym.envs.wrappers.simple_minigrid import SimpleObsWrapper


logger = logging.getLogger(__name__)


@dataclass
class Gym(EnvWrapper):
    env_name: str
    max_steps: Optional[int] = None

    def make(self) -> gym.Env:
        kwargs = {}
        if self.max_steps is not None:
            kwargs["max_steps"] = self.max_steps
        env: gym.Env = gym.make(self.env_name, **kwargs)
        if self.env_name.startswith("MiniGrid-"):
            # Wrap in minigrid simplifier
            env = SimpleObsWrapper(ReseedWrapper(env))

        logger.info(
            f"Env: {self.env_name}; observation_space: {env.observation_space}; "
            f"action_space: {env.action_space}"
        )
        return env
