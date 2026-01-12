#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging
import random
from typing import Dict, List, Optional

import gym
import numpy as np
import pandas as pd
import torch  # @manual
import torch.nn.functional as F
from gym import spaces
from reagent.core.parameters import NormalizationData, NormalizationKey, ProblemDomain
from reagent.gym.agents.agent import Agent
from reagent.gym.agents.post_step import add_replay_buffer_post_step
from reagent.gym.envs import EnvWrapper
from reagent.gym.normalizers import (
    discrete_action_normalizer,
    only_continuous_action_normalizer,
    only_continuous_normalizer,
)
from reagent.gym.policies.random_policies import make_random_policy_for_env
from reagent.gym.runners.gymrunner import run_episode
from reagent.replay_memory import ReplayBuffer
from tqdm import tqdm


logger = logging.getLogger(__name__)

SEED = 0

try:
    from reagent.gym.envs import RecSim  # noqa

    HAS_RECSIM = True
except ImportError:
    HAS_RECSIM = False


def fill_replay_buffer(
    env, replay_buffer: ReplayBuffer, desired_size: int, agent: Agent
):
    """Fill replay buffer with transitions until size reaches desired_size."""
    assert 0 < desired_size and desired_size <= replay_buffer._replay_capacity, (
        f"It's not true that 0 < {desired_size} <= {replay_buffer._replay_capacity}."
    )
    assert replay_buffer.size < desired_size, (
        f"Replay buffer already has {replay_buffer.size} elements. "
        f"(more than desired_size = {desired_size})"
    )
    logger.info(
        f" Starting to fill replay buffer using policy to size: {desired_size}."
    )
    post_step = add_replay_buffer_post_step(replay_buffer, env=env)
    agent.post_transition_callback = post_step

    max_episode_steps = env.max_steps
    with tqdm(
        total=desired_size - replay_buffer.size,
        desc=f"Filling replay buffer from {replay_buffer.size} to size {desired_size}",
    ) as pbar:
        mdp_id = 0
        while replay_buffer.size < desired_size:
            last_size = replay_buffer.size
            max_steps = desired_size - replay_buffer.size
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
    if isinstance(env.observation_space, spaces.Box):
        assert len(env.observation_space.shape) == 1, (
            f"{env.observation_space.shape} has dim > 1, and is not supported."
        )
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
    action_space = env.action_space
    if isinstance(action_space, spaces.Discrete):
        return discrete_action_normalizer(list(range(action_space.n)))
    elif isinstance(action_space, spaces.Box):
        assert len(action_space.shape) == 1, (
            f"Box action shape {action_space.shape} not supported."
        )

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
        return env.normalization_data
    except AttributeError:
        # TODO: make this a property of EnvWrapper?
        if HAS_RECSIM and isinstance(env, RecSim):
            return {
                NormalizationKey.STATE: NormalizationData(
                    dense_normalization_parameters=only_continuous_normalizer(
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


def create_df_from_replay_buffer(
    env,
    problem_domain: ProblemDomain,
    desired_size: int,
    multi_steps: Optional[int],
    ds: str,
    shuffle_df: bool = True,
) -> pd.DataFrame:
    # fill the replay buffer
    set_seed(env, SEED)
    if multi_steps is None:
        update_horizon = 1
        return_as_timeline_format = False
    else:
        update_horizon = multi_steps
        return_as_timeline_format = True
    is_multi_steps = multi_steps is not None

    # The last element of replay buffer always lacks
    # next_action and next_possible_actions.
    # To get full data for every returned sample, we create
    # replay buffer of desired_size + 1 and discard the last element.
    replay_buffer = ReplayBuffer(
        replay_capacity=desired_size + 1,
        batch_size=1,
        update_horizon=update_horizon,
        return_as_timeline_format=return_as_timeline_format,
    )
    random_policy = make_random_policy_for_env(env)
    agent = Agent.create_for_env(env, policy=random_policy)
    fill_replay_buffer(env, replay_buffer, desired_size + 1, agent)

    batch = replay_buffer.sample_transition_batch(
        batch_size=desired_size, indices=torch.arange(desired_size)
    )
    n = batch.state.shape[0]
    logger.info(f"Creating df of size {n}.")

    def discrete_feat_transform(elem) -> str:
        """query data expects str format"""
        return str(elem.item())

    def continuous_feat_transform(elem: List[float]) -> Dict[int, float]:
        """query data expects sparse format"""
        assert isinstance(elem, torch.Tensor), f"{type(elem)} isn't tensor"
        assert len(elem.shape) == 1, f"{elem.shape} isn't 1-dimensional"
        return {i: s.item() for i, s in enumerate(elem)}

    def make_parametric_feat_transform(one_hot_dim: int):
        """one-hot and then continuous_feat_transform"""

        def transform(elem) -> Dict[int, float]:
            elem_tensor = torch.tensor(elem.item())
            one_hot_feat = F.one_hot(elem_tensor, one_hot_dim).float()
            # pyre-fixme[6]: For 1st argument expected `List[float]` but got `Tensor`.
            return continuous_feat_transform(one_hot_feat)

        return transform

    state_features = feature_transform(batch.state, continuous_feat_transform)
    next_state_features = feature_transform(
        batch.next_state,
        continuous_feat_transform,
        is_next_with_multi_steps=is_multi_steps,
    )

    if problem_domain == ProblemDomain.DISCRETE_ACTION:
        # discrete action is str
        action = feature_transform(batch.action, discrete_feat_transform)
        next_action = feature_transform(
            batch.next_action,
            discrete_feat_transform,
            is_next_with_multi_steps=is_multi_steps,
            replace_when_terminal="",
            terminal=batch.terminal,
        )
    elif problem_domain == ProblemDomain.PARAMETRIC_ACTION:
        # continuous action is Dict[int, double]
        assert isinstance(env.action_space, gym.spaces.Discrete)
        parametric_feat_transform = make_parametric_feat_transform(env.action_space.n)
        action = feature_transform(batch.action, parametric_feat_transform)
        next_action = feature_transform(
            batch.next_action,
            parametric_feat_transform,
            is_next_with_multi_steps=is_multi_steps,
            replace_when_terminal={},
            terminal=batch.terminal,
        )
    elif problem_domain == ProblemDomain.CONTINUOUS_ACTION:
        action = feature_transform(batch.action, continuous_feat_transform)
        next_action = feature_transform(
            batch.next_action,
            continuous_feat_transform,
            is_next_with_multi_steps=is_multi_steps,
            replace_when_terminal={},
            terminal=batch.terminal,
        )
    elif problem_domain == ProblemDomain.MDN_RNN:
        action = feature_transform(batch.action, discrete_feat_transform)
        assert multi_steps is not None
        next_action = feature_transform(
            batch.next_action,
            discrete_feat_transform,
            is_next_with_multi_steps=True,
            replace_when_terminal="",
            terminal=batch.terminal,
        )
    else:
        raise NotImplementedError(f"model type: {problem_domain}.")

    if multi_steps is None:
        time_diff = [1] * n
        reward = batch.reward.squeeze(1).tolist()
        metrics = [{"reward": r} for r in reward]
    else:
        time_diff = [[1] * len(ns) for ns in next_state_features]
        reward = [reward_list.tolist() for reward_list in batch.reward]
        metrics = [
            [{"reward": r.item()} for r in reward_list] for reward_list in batch.reward
        ]

    # TODO(T67265031): change this to int
    mdp_id = [str(i.item()) for i in batch.mdp_id]
    sequence_number = batch.sequence_number.squeeze(1).tolist()
    # in the product data, all sequence_number_ordinal start from 1.
    # So to be consistent with the product data.

    sequence_number_ordinal = (batch.sequence_number.squeeze(1) + 1).tolist()
    action_probability = batch.log_prob.exp().squeeze(1).tolist()
    df_dict = {
        "state_features": state_features,
        "next_state_features": next_state_features,
        "action": action,
        "next_action": next_action,
        "reward": reward,
        "action_probability": action_probability,
        "metrics": metrics,
        "time_diff": time_diff,
        "mdp_id": mdp_id,
        "sequence_number": sequence_number,
        "sequence_number_ordinal": sequence_number_ordinal,
        "ds": [ds] * n,
    }

    if problem_domain == ProblemDomain.PARAMETRIC_ACTION:
        # Possible actions are List[Dict[int, float]]
        assert isinstance(env.action_space, gym.spaces.Discrete)
        possible_actions = [{i: 1.0} for i in range(env.action_space.n)]

    elif problem_domain == ProblemDomain.DISCRETE_ACTION:
        # Possible actions are List[str]
        assert isinstance(env.action_space, gym.spaces.Discrete)
        possible_actions = [str(i) for i in range(env.action_space.n)]

    elif problem_domain == ProblemDomain.MDN_RNN:
        # Possible actions are List[str]
        assert isinstance(env.action_space, gym.spaces.Discrete)
        possible_actions = [str(i) for i in range(env.action_space.n)]

    # these are fillers, which should have correct shape
    pa_features = range(n)
    pna_features = time_diff
    if problem_domain in (
        ProblemDomain.DISCRETE_ACTION,
        ProblemDomain.PARAMETRIC_ACTION,
        ProblemDomain.MDN_RNN,
    ):

        def pa_transform(x):
            return possible_actions

        df_dict["possible_actions"] = feature_transform(pa_features, pa_transform)
        df_dict["possible_next_actions"] = feature_transform(
            pna_features,
            pa_transform,
            is_next_with_multi_steps=is_multi_steps,
            replace_when_terminal=[],
            terminal=batch.terminal,
        )

    df = pd.DataFrame(df_dict)
    # validate df
    validate_mdp_ids_seq_nums(df)
    if shuffle_df:
        # shuffling (sample the whole batch)
        df = df.reindex(np.random.permutation(df.index))
    return df


def set_seed(env: gym.Env, seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)


def feature_transform(
    features,
    single_elem_transform,
    is_next_with_multi_steps=False,
    replace_when_terminal=None,
    terminal=None,
):
    """feature_transform is a method on a single row.
    We assume features is List[features] (batch of features).
    This can also be called for next_features with multi_steps which we assume
    to be List[List[features]]. First List is denoting that it's a batch,
    second List is denoting that a single row consists of a list of features.
    """
    if is_next_with_multi_steps:
        if terminal is None:
            return [
                [single_elem_transform(feat) for feat in multi_steps_features]
                for multi_steps_features in features
            ]
        else:
            # for next features where we replace them when terminal
            assert replace_when_terminal is not None
            return [
                (
                    [single_elem_transform(feat) for feat in multi_steps_features]
                    if not terminal[idx]
                    else [
                        single_elem_transform(feat)
                        for feat in multi_steps_features[:-1]
                    ]
                    + [replace_when_terminal]
                )
                for idx, multi_steps_features in enumerate(features)
            ]
    else:
        if terminal is None:
            return [single_elem_transform(feat) for feat in features]
        else:
            assert replace_when_terminal is not None
            return [
                (
                    single_elem_transform(feat)
                    if not terminal[idx]
                    else replace_when_terminal
                )
                for idx, feat in enumerate(features)
            ]


def validate_mdp_ids_seq_nums(df):
    mdp_ids = list(df["mdp_id"])
    sequence_numbers = list(df["sequence_number"])
    unique_mdp_ids = set(mdp_ids)
    prev_mdp_id, prev_seq_num = None, None
    mdp_count = 0
    for mdp_id, seq_num in zip(mdp_ids, sequence_numbers):
        if prev_mdp_id is None or mdp_id != prev_mdp_id:
            mdp_count += 1
            prev_mdp_id = mdp_id
        else:
            assert seq_num == prev_seq_num + 1, (
                f"For mdp_id {mdp_id}, got {seq_num} <= {prev_seq_num}."
                f"Sequence number must be in increasing order.\n"
                f"Zip(mdp_id, seq_num): "
                f"{list(zip(mdp_ids, sequence_numbers))}"
            )
        prev_seq_num = seq_num

    assert len(unique_mdp_ids) == mdp_count, "MDPs are broken up. {} vs {}".format(
        len(unique_mdp_ids), mdp_count
    )
    return
