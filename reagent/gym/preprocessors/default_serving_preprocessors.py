#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

""" Returns preprocessors for serving module inference. """

from typing import Tuple

import numpy as np
import reagent.types as rlt
import torch
from gym import Env, spaces
from reagent.gym.preprocessors.default_preprocessors import discrete_action_extractor


def make_default_serving_obs_preprocessor(env: Env):
    if not isinstance(env.observation_space, spaces.Box):
        raise NotImplementedError(f"{env.observation_space} not supported!")

    observation_space = env.observation_space
    if len(observation_space.shape) != 1:
        raise NotImplementedError(f"Box shape {observation_space.shape} not supported!")

    state_dim = observation_space.shape[0]

    def gym_to_reagent_serving(obs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = torch.tensor(obs).float().view(1, state_dim)
        presence_tensor = torch.ones_like(obs_tensor)
        return (obs_tensor, presence_tensor)

    return gym_to_reagent_serving


def make_default_serving_action_extractor(env: Env):
    if isinstance(env.action_space, spaces.Discrete):
        return discrete_action_extractor
    elif isinstance(env.action_space, spaces.Box):
        assert (
            len(env.action_space.shape) == 1
        ), f"Unsupported Box with shape {env.action_space.shape}"
        return continuous_predictor_action_extractor
    else:
        raise NotImplementedError


def continuous_predictor_action_extractor(output: rlt.ActorOutput):
    assert (
        len(output.action.shape) == 2 and output.action.shape[0] == 1
    ), f"{output.action.shape} isn't (1, action_dim)"
    return output.action.squeeze(0).cpu().numpy()
