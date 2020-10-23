#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict

# pyre-fixme[21]: Could not find module `gym`.
from gym import spaces
from reagent.gym.agents.agent import Agent
from reagent.gym.agents.post_step import add_replay_buffer_post_step
from reagent.gym.envs import EnvWrapper
from reagent.gym.policies.random_policies import make_random_policy_for_env
from reagent.gym.runners.gymrunner import run_episode
from reagent.parameters import NormalizationData, NormalizationKey
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.test.base.utils import (
    only_continuous_action_normalizer,
    only_continuous_normalizer,
)

# pyre-fixme[21]: Could not find module `tqdm`.
from tqdm import tqdm


logger = logging.getLogger(__name__)

try:
    from reagent.gym.envs import RecSim  # noqa

    HAS_RECSIM = True
except ImportError:
    HAS_RECSIM = False


def fill_replay_buffer(env, replay_buffer: ReplayBuffer, desired_size: int):
    """ Fill replay buffer with random transitions until size reaches desired_size. """
    assert (
        0 < desired_size and desired_size <= replay_buffer._replay_capacity
    ), f"It's not true that 0 < {desired_size} <= {replay_buffer._replay_capacity}."
    assert replay_buffer.size < desired_size, (
        f"Replay buffer already has {replay_buffer.size} elements. "
        f"(more than desired_size = {desired_size})"
    )
    logger.info(f"Starting to fill replay buffer to size: {desired_size}.")
    random_policy = make_random_policy_for_env(env)
    post_step = add_replay_buffer_post_step(replay_buffer, env=env)
    agent = Agent.create_for_env(
        env, policy=random_policy, post_transition_callback=post_step
    )
    max_episode_steps = env.max_steps
    with tqdm(
        total=desired_size - replay_buffer.size,
        desc=f"Filling replay buffer from {replay_buffer.size} to size {desired_size}",
    ) as pbar:
        mdp_id = 0
        while replay_buffer.size < desired_size:
            last_size = replay_buffer.size
            max_steps = desired_size - replay_buffer.size - 1
            if max_episode_steps is not None:
                max_steps = min(max_episode_steps, max_steps)
            run_episode(env=env, agent=agent, mdp_id=mdp_id, max_steps=max_steps)
            size_delta = replay_buffer.size - last_size
            # The assertion below is commented out because it can't
            # support input samples which has seq_len>1. This should be
            # treated as a bug, and need to be fixed in the future.
            # assert (
            #     size_delta >= 0
            # ), f"size delta is {size_delta} which should be non-negative."
            pbar.update(n=size_delta)
            mdp_id += 1
            if size_delta <= 0:
                # replay buffer size isn't increasing... so stop early
                break

    if replay_buffer.size >= desired_size:
        logger.info(f"Successfully filled replay buffer to size: {replay_buffer.size}!")
    else:
        logger.info(
            f"Stopped early and filled replay buffer to size: {replay_buffer.size}."
        )


def build_state_normalizer(env: EnvWrapper):
    # pyre-fixme[16]: `EnvWrapper` has no attribute `observation_space`.
    if isinstance(env.observation_space, spaces.Box):
        assert (
            len(env.observation_space.shape) == 1
        ), f"{env.observation_space.shape} has dim > 1, and is not supported."
        return only_continuous_normalizer(
            list(range(env.observation_space.shape[0])),
            env.observation_space.low,
            env.observation_space.high,
        )
    elif isinstance(env.observation_space, spaces.Dict):
        # assuming env.observation_space is image
        return None
    else:
        raise NotImplementedError(f"{env.observation_space} not supported")


def build_action_normalizer(env: EnvWrapper):
    # pyre-fixme[16]: `EnvWrapper` has no attribute `action_space`.
    action_space = env.action_space
    if isinstance(action_space, spaces.Discrete):
        return only_continuous_normalizer(
            list(range(action_space.n)), min_value=0, max_value=1
        )
    elif isinstance(action_space, spaces.Box):
        assert (
            len(action_space.shape) == 1
        ), f"Box action shape {action_space.shape} not supported."

        action_dim = action_space.shape[0]
        return only_continuous_action_normalizer(
            list(range(action_dim)),
            min_value=action_space.low,
            max_value=action_space.high,
        )
    else:
        raise NotImplementedError(f"{action_space} not supported.")


def build_normalizer(env: EnvWrapper) -> Dict[str, NormalizationData]:
    try:
        # pyre-fixme[16]: `EnvWrapper` has no attribute `normalization_data`.
        return env.normalization_data
    except AttributeError:
        # TODO: make this a property of EnvWrapper?
        # pyre-fixme[16]: Module `envs` has no attribute `RecSim`.
        if HAS_RECSIM and isinstance(env, RecSim):
            return {
                NormalizationKey.STATE: NormalizationData(
                    dense_normalization_parameters=only_continuous_normalizer(
                        # pyre-fixme[16]: `RecSim` has no attribute `observation_space`.
                        list(range(env.observation_space["user"].shape[0]))
                    )
                ),
                NormalizationKey.ITEM: NormalizationData(
                    dense_normalization_parameters=only_continuous_normalizer(
                        list(range(env.observation_space["doc"]["0"].shape[0]))
                    )
                ),
            }
        return {
            NormalizationKey.STATE: NormalizationData(
                dense_normalization_parameters=build_state_normalizer(env)
            ),
            NormalizationKey.ACTION: NormalizationData(
                dense_normalization_parameters=build_action_normalizer(env)
            ),
        }
