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
from reagent.core.types import TableSpec
from reagent.gym.agents.agent import Agent
from reagent.gym.envs.gym import Gym
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model
from reagent.gym.runners.gymrunner import evaluate_for_n_episodes
from reagent.gym.utils import fill_replay_buffer
from reagent.publishers.union import FileSystemPublisher, ModelPublisher__Union
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.replay_memory.utils import replay_buffer_to_pre_timeline_df
from reagent.workflow.model_managers.union import ModelManager__Union
from reagent.workflow.spark_utils import call_spark_class, get_spark_session


logger = logging.getLogger(__name__)


def initialize_seed(seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def offline_gym(
    env_name: str,
    pkl_path: str,
    num_train_transitions: int,
    max_steps: Optional[int],
    seed: Optional[int] = None,
):
    """
    Generate samples from a DiscreteRandomPolicy on the Gym environment and
    saves results in a pandas df parquet.
    """
    initialize_seed(seed)
    env = Gym(env_name=env_name)

    replay_buffer = ReplayBuffer(replay_capacity=num_train_transitions, batch_size=1)
    fill_replay_buffer(env, replay_buffer, num_train_transitions)
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
    """ Loads a pandas parquet, converts to pyspark, and uploads df to Hive.
        Then call the timeline operator.
    """

    pd_df = pd.read_pickle(pkl_path)
    spark = get_spark_session()
    df = spark.createDataFrame(pd_df)
    input_name = f"{input_table_spec.table}{PRE_TIMELINE_SUFFIX}"
    df.createTempView(input_name)

    output_name = input_table_spec.table
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


def evaluate_gym(
    env_name: str,
    model: ModelManager__Union,
    publisher: ModelPublisher__Union,
    num_eval_episodes: int,
    passing_score_bar: float,
    max_steps: Optional[int] = None,
):
    publisher_manager = publisher.value
    assert isinstance(
        publisher_manager, FileSystemPublisher
    ), f"publishing manager is type {type(publisher_manager)}, not FileSystemPublisher"
    env = Gym(env_name=env_name)
    torchscript_path = publisher_manager.get_latest_published_model(model.value)
    jit_model = torch.jit.load(torchscript_path)
    policy = create_predictor_policy_from_model(jit_model)
    agent = Agent.create_for_env_with_serving_policy(env, policy)
    rewards = evaluate_for_n_episodes(
        n=num_eval_episodes, env=env, agent=agent, max_steps=max_steps
    )
    avg_reward = np.mean(rewards)
    logger.info(
        f"Average reward over {num_eval_episodes} is {avg_reward}.\n"
        f"List of rewards: {rewards}"
    )
    assert (
        avg_reward >= passing_score_bar
    ), f"{avg_reward} fails to pass the bar of {passing_score_bar}!"
    return
