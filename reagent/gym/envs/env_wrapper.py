#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import abc

import gym
from reagent.core.dataclasses import dataclass
from reagent.core.registry_meta import RegistryMeta


@dataclass
class EnvWrapper(gym.core.Wrapper, metaclass=RegistryMeta):
    """ Wrapper around it's environment, to simplify configuration. """

    def __post_init_post_parse__(self):
        super().__init__(self.make())

    def __getattr__(self, attr):
        raise AttributeError(f"Trying to get {attr}")

    @abc.abstractmethod
    def make(self) -> gym.Env:
        pass

    # TODO: add more methods to simplify gym code
    # e.g. normalization, specific preprocessor, etc.
    # This can move a lot of the if statements from create_from_env methods.
