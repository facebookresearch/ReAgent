#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import copy
import logging
from enum import Enum

import gym
import gym.spaces.dict
import numpy as np


logger = logging.getLogger(__name__)


class ValueMode(Enum):
    CONST = 0
    INNER_PROD = 1


class ValueWrapper(gym.core.ObservationWrapper):
    KEY = "value"

    def __init__(self, env, value_mode: ValueMode):
        super().__init__(env)
        self.value_mode = value_mode

    @property
    def observation_space(self):
        obs_spaces = copy.copy(self.env.observation_space.spaces)
        try:
            augmentation = obs_spaces["augmentation"]
        except KeyError:
            augmentation = gym.spaces.Dict()
            obs_spaces["augmentation"] = augmentation

        for k in obs_spaces["doc"].spaces:
            try:
                aug_k = augmentation[k]
            except KeyError:
                aug_k = gym.spaces.Dict()
                augmentation.spaces[k] = aug_k

            assert not aug_k.contains(self.KEY)

            aug_k.spaces[self.KEY] = gym.spaces.Box(low=-1.0, high=1.0, shape=())

        return gym.spaces.Dict(obs_spaces)

    @observation_space.setter
    def observation_space(self, x):
        # We just have this method here so that Wrapper.__init__() can run
        pass

    def observation(self, obs):
        try:
            augmentation = obs["augmentation"]
        except KeyError:
            augmentation = {}
            obs["augmentation"] = augmentation

        for k in obs["doc"]:
            try:
                aug_k = augmentation[k]
            except KeyError:
                aug_k = {}
                augmentation[k] = aug_k

            if self.value_mode == ValueMode.CONST:
                aug_k[self.KEY] = 0.0
            elif self.value_mode == ValueMode.INNER_PROD:
                aug_k[self.KEY] = np.inner(obs["user"], obs["doc"][k])
            else:
                raise NotImplementedError(f"{self.value_mode} is not implemented")

        return obs
