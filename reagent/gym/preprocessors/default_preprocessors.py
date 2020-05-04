#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

""" Get default preprocessors for training time. """

import numpy as np
import reagent.types as rlt
import torch
from gym import Env, spaces
from reagent.parameters import CONTINUOUS_TRAINING_ACTION_RANGE


def make_default_obs_preprocessor(env: Env):
    """ Returns the default obs preprocessor for the environment """
    observation_space = env.observation_space
    if isinstance(observation_space, spaces.Box):
        return box_obs_preprocessor
    else:
        raise NotImplementedError(f"Unsupport observation space: {observation_space}")


def make_default_action_extractor(env: Env):
    """ Returns the default action extractor for the environment """
    action_space = env.action_space
    if isinstance(action_space, spaces.Discrete):
        return discrete_action_extractor
    elif isinstance(action_space, spaces.Box):
        return make_box_action_extractor(action_space)
    else:
        raise NotImplementedError(f"Unsupport action space: {action_space}")


#######################################
### Default obs preprocessors.
### These should operate on single obs.
#######################################
def box_obs_preprocessor(obs: torch.Tensor) -> rlt.PreprocessedState:
    return rlt.PreprocessedState.from_tensor(obs.float()).unsqueeze(0)


############################################
### Default action extractors.
### These currently operate on single action.
############################################
def discrete_action_extractor(actor_output: rlt.ActorOutput):
    action = actor_output.action
    assert (
        action.ndim == 2 and action.shape[0] == 1
    ), f"{action} is not a single batch of results!"
    return action.squeeze(0).argmax().cpu().numpy()


def rescale_actions(
    actions: np.ndarray,
    new_min: float,
    new_max: float,
    prev_min: float,
    prev_max: float,
):
    """ Scale from [prev_min, prev_max] to [new_min, new_max] """
    assert np.all(prev_min <= actions) and np.all(
        actions <= prev_max
    ), f"{actions} has values outside of [{prev_min}, {prev_max}]."
    prev_range = prev_max - prev_min
    new_range = new_max - new_min
    return ((actions - prev_min) / prev_range) * new_range + new_min


def make_box_action_extractor(action_space: spaces.Box):
    assert (
        len(action_space.shape) == 1 and action_space.shape[0] == 1
    ), f"{action_space} not supported."

    model_low, model_high = CONTINUOUS_TRAINING_ACTION_RANGE
    env_low = action_space.low.item()
    env_high = action_space.high.item()

    def box_action_extractor(actor_output: rlt.ActorOutput):
        action = actor_output.action
        assert (
            action.ndim == 2 and action.shape[0] == 1
        ), f"{action} is not a single batch of results!"
        return rescale_actions(
            action.squeeze(0).cpu().numpy(),
            new_min=env_low,
            new_max=env_high,
            prev_min=model_low,
            prev_max=model_high,
        )

    return box_action_extractor
