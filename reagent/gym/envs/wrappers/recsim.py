#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import copy
import logging

import gym
import gym.spaces.dict


logger = logging.getLogger(__name__)


class ValueWrapper(gym.core.ObservationWrapper):
    KEY = "value"

    def __init__(self, env, value_fn):
        """
        Args:
          env: a RecSim gym environment
          value_fn: a function taking user & document feature,
            returning the value of the document for the user
        """
        super().__init__(env)
        self.value_fn = value_fn

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

            aug_k[self.KEY] = self.value_fn(obs["user"], obs["doc"][k])

        return obs
