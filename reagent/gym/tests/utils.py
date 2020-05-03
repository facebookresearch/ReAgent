#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from gym import spaces
from reagent.parameters import NormalizationData, NormalizationKey
from reagent.test.base.utils import (
    only_continuous_action_normalizer,
    only_continuous_normalizer,
)


def build_state_normalizer(env):
    if isinstance(env.observation_space, spaces.Box):
        assert (
            len(env.observation_space.shape) == 1
        ), f"{env.observation_space.shape} has dim > 1, and is not supported."
        return NormalizationData(
            dense_normalization_parameters=only_continuous_normalizer(
                list(range(env.observation_space.shape[0])),
                env.observation_space.low,
                env.observation_space.high,
            )
        )
    elif isinstance(env.observation_space, spaces.Dict):
        # assuming env.observation_space is image
        return None
    else:
        raise NotImplementedError(f"{env.observation_space} not supported")


def build_action_normalizer(env):
    action_space = env.action_space
    if isinstance(action_space, spaces.Discrete):
        return only_continuous_normalizer(
            list(range(action_space.n)), min_value=0, max_value=1
        )
    elif isinstance(action_space, spaces.Box):
        assert action_space.shape == (
            1,
        ), f"Box action shape {action_space.shape} not supported."

        return NormalizationData(
            dense_normalization_parameters=only_continuous_action_normalizer(
                [0],
                min_value=action_space.low.item(),
                max_value=action_space.high.item(),
            )
        )
    else:
        raise NotImplementedError(f"{action_space} not supported.")


def build_normalizer(env):
    return {
        NormalizationKey.STATE: build_state_normalizer(env),
        NormalizationKey.ACTION: build_action_normalizer(env),
    }
