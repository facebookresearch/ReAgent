#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import abc
import logging
from typing import Callable, Optional

# pyre-fixme[21]: Could not find module `gym`.
import gym
import numpy as np
import reagent.types as rlt
import torch
from gym import spaces
from reagent.core.dataclasses import dataclass
from reagent.core.registry_meta import RegistryMeta
from reagent.parameters import CONTINUOUS_TRAINING_ACTION_RANGE
from reagent.training.utils import rescale_actions


# types for reference
ObsPreprocessor = Callable[[np.ndarray], rlt.FeatureData]
ServingObsPreprocessor = Callable[[np.ndarray], rlt.ServingFeatureData]
ActionExtractor = Callable[[rlt.ActorOutput], np.ndarray]
ServingActionExtractor = ActionExtractor

CONTINUOUS_MODEL_LOW = torch.tensor(CONTINUOUS_TRAINING_ACTION_RANGE[0])
CONTINUOUS_MODEL_HIGH = torch.tensor(CONTINUOUS_TRAINING_ACTION_RANGE[1])

logger = logging.getLogger(__name__)


@dataclass
# pyre-fixme[11]: Annotation `Wrapper` is not defined as a type.
class EnvWrapper(gym.core.Wrapper, metaclass=RegistryMeta):
    """ Wrapper around it's environment, to simplify configuration. """

    def __post_init_post_parse__(self):
        super().__init__(self.make())
        logger.info(
            f"Env: {self.env};\n"
            f"observation_space: {self.env.observation_space};\n"
            f"action_space: {self.env.action_space};"
        )

    @abc.abstractmethod
    # pyre-fixme[11]: Annotation `Env` is not defined as a type.
    def make(self) -> gym.Env:
        pass

    @abc.abstractmethod
    def obs_preprocessor(self, obs: np.ndarray) -> rlt.FeatureData:
        pass

    @abc.abstractmethod
    def serving_obs_preprocessor(self, obs: np.ndarray) -> rlt.ServingFeatureData:
        pass

    def get_obs_preprocessor(self, *ctor_args, **ctor_kwargs):
        # ctor_args go to .to call
        ctor_kwargs["non_blocking"] = True
        return lambda *args, **kwargs: self.obs_preprocessor(*args, **kwargs).to(
            *ctor_args, **ctor_kwargs
        )

    def get_serving_obs_preprocessor(self):
        return lambda *args, **kwargs: self.serving_obs_preprocessor(*args, **kwargs)

    def action_extractor(self, actor_output: rlt.ActorOutput) -> torch.Tensor:
        action = actor_output.action
        # pyre-fixme[16]: `EnvWrapper` has no attribute `action_space`.
        action_space = self.action_space
        # Canonical rule to return one-hot encoded actions for discrete
        assert (
            len(action.shape) == 2 and action.shape[0] == 1
        ), f"{action} (shape: {action.shape}) is not a single action!"
        if isinstance(action_space, spaces.Discrete):
            # pyre-fixme[16]: `Tensor` has no attribute `argmax`.
            return action.squeeze(0).argmax()
        elif isinstance(action_space, spaces.MultiDiscrete):
            return action.squeeze(0)
        # Canonical rule to scale actions to CONTINUOUS_TRAINING_ACTION_RANGE
        elif isinstance(action_space, spaces.Box):
            assert len(action_space.shape) == 1, f"{action_space} not supported."
            return rescale_actions(
                action.squeeze(0),
                new_min=torch.tensor(action_space.low),
                new_max=torch.tensor(action_space.high),
                prev_min=CONTINUOUS_MODEL_LOW,
                prev_max=CONTINUOUS_MODEL_HIGH,
            )
        else:
            raise NotImplementedError(f"Unsupported action space: {action_space}")

    def serving_action_extractor(self, actor_output: rlt.ActorOutput) -> torch.Tensor:
        action = actor_output.action
        # pyre-fixme[16]: `EnvWrapper` has no attribute `action_space`.
        action_space = self.action_space
        assert (
            len(action.shape) == 2 and action.shape[0] == 1
        ), f"{action.shape} isn't (1, action_dim)"
        if isinstance(action_space, spaces.Discrete):
            # pyre-fixme[16]: `Tensor` has no attribute `argmax`.
            return action.squeeze(0).argmax().view([])
        elif isinstance(action_space, spaces.MultiDiscrete):
            return action.squeeze(0)
        elif isinstance(action_space, spaces.Box):
            assert (
                len(action_space.shape) == 1
            ), f"Unsupported Box with shape {action_space.shape}"
            return action.squeeze(0)
        else:
            raise NotImplementedError(f"Unsupported action space: {action_space}")

    def get_action_extractor(self):
        return (
            lambda *args, **kwargs: self.action_extractor(*args, **kwargs).cpu().numpy()
        )

    def get_serving_action_extractor(self):
        return (
            lambda *args, **kwargs: self.serving_action_extractor(*args, **kwargs)
            .cpu()
            .numpy()
        )

    # TODO: add more methods to simplify gym code
    # e.g. normalization, specific preprocessor, etc.
    # This can move a lot of the if statements from create_from_env methods.

    @property
    def max_steps(self) -> Optional[int]:
        possible_keys = [
            # gym should have _max_episode_steps
            "_max_episode_steps",
            # Minigrid should have max_steps
            "max_steps",
        ]
        for key in possible_keys:
            # pyre-fixme[16]: `EnvWrapper` has no attribute `env`.
            res = getattr(self.env, key, None)
            if res is not None:
                return res
        return None

    @property
    def possible_actions_mask(self) -> Optional[np.ndarray]:
        # pyre-fixme[16]: `EnvWrapper` has no attribute `env`.
        return getattr(self.env, "possible_actions_mask", None)
