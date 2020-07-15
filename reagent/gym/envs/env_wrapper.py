#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import abc
import logging

import gym
from reagent.core.dataclasses import dataclass
from reagent.core.registry_meta import RegistryMeta


logger = logging.getLogger(__name__)


@dataclass
class EnvWrapper(gym.core.Wrapper, metaclass=RegistryMeta):
    """ Wrapper around it's environment, to simplify configuration. """

    def __post_init_post_parse__(self):
        super().__init__(self.make())
        logger.info(
            f"Env: {self.env};\n"
            f"observation_space: {self.env.observation_space};\n"
            f"action_space: {self.env.action_space};"
        )

    def __getattr__(self, attr):
        raise AttributeError(f"Trying to get {attr}")

    @abc.abstractmethod
    def make(self) -> gym.Env:
        pass

    # TODO: add more methods to simplify gym code
    # e.g. normalization, specific preprocessor, etc.
    # This can move a lot of the if statements from create_from_env methods.
