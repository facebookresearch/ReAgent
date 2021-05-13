#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import json
import logging
import random
from typing import Optional

import gym
import numpy as np
import pandas as pd
import torch
from reagent.data.spark_utils import call_spark_class, get_spark_session
from reagent.gym.agents.agent import Agent
from reagent.gym.envs import Gym
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.gym.policies.random_policies import make_random_policy_for_env
from reagent.gym.runners.gymrunner import evaluate_for_n_episodes
from reagent.gym.utils import fill_replay_buffer
from reagent.model_managers.union import ModelManager__Union
from reagent.publishers.union import FileSystemPublisher, ModelPublisher__Union
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.replay_memory.utils import replay_buffer_to_pre_timeline_df

from .types import TableSpec


logger = logging.getLogger(__name__)


def initialize_seed(seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def offline_gym_random(
    env_name: str,
    pkl_path: str,
    num_train_transitions: int,
    max_steps: Optional[int],
    seed: int = 1,
):
    """
    Generate samples from a random Policy on the Gym environment and
    saves results in a pandas df parquet.
    """
    env = Gym(env_name=env_name)
    random_policy = make_random_policy_for_env(env)
    agent = Agent.create_for_env(env, policy=random_policy)
    return _offline_gym(env, agent, pkl_path, num_train_transitions, max_steps, seed)


def offline_gym_predictor(
    env_name: str,
    model: ModelManager__Union,
    publisher: ModelPublisher__Union,
    pkl_path: str,
    num_train_transitions: int,
    max_steps: Optional[int],
    module_name: str = "default_model",
    seed: int = 1,
):
    """
    Generate samples from a trained Policy on the Gym environment and
    saves results in a pandas df parquet.
    """
    env = Gym(env_name=env_name)
    agent = make_agent_from_model(env, model, publisher, module_name)
    return _offline_gym(env, agent, pkl_path, num_train_transitions, max_steps, seed)


def _offline_gym(
    env: Gym,
    agent: Agent,
    pkl_path: str,
    num_train_transitions: int,
    max_steps: Optional[int],
    seed: int = 1,
):
    initialize_seed(seed)

    replay_buffer = ReplayBuffer(replay_capacity=num_train_transitions, batch_size=1)
    fill_replay_buffer(env, replay_buffer, num_train_transitions, agent)
    if isinstance(env.action_space, gym.spaces.Discrete):
        is_discrete_action = True
    else:
        assert isinstance(env.action_space, gym.spaces.Box)
        is_discrete_action = False
    df = replay_buffer_to_pre_timeline_df(is_discrete_action, replay_buffer)
    logger.info(f"Saving dataset with {len(df)} samples to {pkl_path}")
    df.to_pickle(pkl_path)


PRE_TIMELINE_SUFFIX = "_pre_timeline_operator"


def timeline_operator(pkl_path: str, input_table_spec: TableSpec):
    """Loads a pandas parquet, converts to pyspark, and uploads df to Hive.
    Then call the timeline operator.
    """

    pd_df = pd.read_pickle(pkl_path)
    spark = get_spark_session()
    df = spark.createDataFrame(pd_df)
    input_name = f"{input_table_spec.table_name}{PRE_TIMELINE_SUFFIX}"
    df.createTempView(input_name)

    output_name = input_table_spec.table_name
    include_possible_actions = "possible_actions" in pd_df
    arg = {
        "startDs": "2019-01-01",
        "endDs": "2019-01-01",
        "addTerminalStateRow": True,
        "inputTableName": input_name,
        "outputTableName": output_name,
        "includePossibleActions": include_possible_actions,
        "percentileFunction": "percentile_approx",
        "rewardColumns": ["reward", "metrics"],
        "extraFeatureColumns": [],
    }
    call_spark_class(spark, class_name="Timeline", args=json.dumps(arg))


def make_agent_from_model(
    env: Gym,
    model: ModelManager__Union,
    publisher: ModelPublisher__Union,
    module_name: str,
):
    publisher_manager = publisher.value
    assert isinstance(
        publisher_manager, FileSystemPublisher
    ), f"publishing manager is type {type(publisher_manager)}, not FileSystemPublisher"
    module_names = model.value.serving_module_names()
    assert module_name in module_names, f"{module_name} not in {module_names}"
    torchscript_path = publisher_manager.get_latest_published_model(
        model.value, module_name
    )
    jit_model = torch.jit.load(torchscript_path)
    policy = create_predictor_policy_from_model(jit_model)
    agent = Agent.create_for_env_with_serving_policy(env, policy)
    return agent


def evaluate_gym(
    env_name: str,
    model: ModelManager__Union,
    publisher: ModelPublisher__Union,
    num_eval_episodes: int,
    passing_score_bar: float,
    module_name: str = "default_model",
    max_steps: Optional[int] = None,
):
    initialize_seed(1)
    env = Gym(env_name=env_name)
    agent = make_agent_from_model(env, model, publisher, module_name)

    rewards = evaluate_for_n_episodes(
        n=num_eval_episodes, env=env, agent=agent, max_steps=max_steps
    )
    avg_reward = np.mean(rewards)
    logger.info(
        f"Average reward over {num_eval_episodes} is {avg_reward}.\n"
        f"List of rewards: {rewards}\n"
        f"Passing score bar: {passing_score_bar}"
    )
    assert (
        avg_reward >= passing_score_bar
    ), f"{avg_reward} fails to pass the bar of {passing_score_bar}!"
    return
