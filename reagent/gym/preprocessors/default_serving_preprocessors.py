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
    def gym_to_reagent_serving(obs: np.array) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = torch.tensor(obs).float().unsqueeze(0)
        presence_tensor = torch.ones_like(obs_tensor)
        return (obs_tensor, presence_tensor)

    return gym_to_reagent_serving


def make_default_serving_action_extractor(env: Env):
    if isinstance(env.action_space, spaces.Discrete):
        return discrete_action_extractor
    elif isinstance(env.action_space, spaces.Box):
        assert env.action_space.shape == (1,)
        return continuous_predictor_action_extractor


def continuous_predictor_action_extractor(output: rlt.ActorOutput):
    action = output.action.item()
    return np.array([action])
