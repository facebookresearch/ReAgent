#!/usr/bin/env python3

import logging
from typing import Tuple

import gym
import numpy as np
import reagent.types as rlt
import torch
from gym import spaces
from gym_minigrid.wrappers import ReseedWrapper
from reagent.core.dataclasses import dataclass
from reagent.gym.envs.env_wrapper import EnvWrapper
from reagent.gym.envs.wrappers.simple_minigrid import SimpleObsWrapper


logger = logging.getLogger(__name__)


@dataclass
class Gym(EnvWrapper):
    env_name: str

    def make(self) -> gym.Env:
        env: gym.Env = gym.make(self.env_name)
        if self.env_name.startswith("MiniGrid-"):
            # Wrap in minigrid simplifier
            env = SimpleObsWrapper(ReseedWrapper(env))
        return env

    def obs_preprocessor(self, obs: np.ndarray) -> rlt.FeatureData:
        obs_space = self.observation_space
        if isinstance(obs_space, spaces.Box):
            return rlt.FeatureData(torch.tensor(obs).float().unsqueeze(0))
        else:
            raise NotImplementedError(f"{obs_space} obs space not supported for Gym.")

    # TODO: make return serving feature data
    # pyre-fixme[15]: `serving_obs_preprocessor` overrides method defined in
    #  `EnvWrapper` inconsistently.
    def serving_obs_preprocessor(
        self, obs: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_space = self.observation_space
        if not isinstance(obs_space, spaces.Box):
            raise NotImplementedError(f"{obs_space} not supported!")

        if len(obs_space.shape) != 1:
            raise NotImplementedError(f"Box shape {obs_space.shape} not supported!")
        state_dim = obs_space.shape[0]
        obs_tensor = torch.tensor(obs).float().view(1, state_dim)
        presence_tensor = torch.ones_like(obs_tensor)
        return (obs_tensor, presence_tensor)
