#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import json
import logging
from typing import Optional

# pyre-fixme[21]: Could not find module `gym`.
import gym
import numpy as np

# pyre-fixme[21]: Could not find module `pandas`.
import pandas as pd

# pyre-fixme[21]: Could not find module `pytorch_lightning`.
import pytorch_lightning as pl
import torch

# pyre-fixme[21]: Could not find module `reagent.data.spark_utils`.
from reagent.data.spark_utils import call_spark_class, get_spark_session

# pyre-fixme[21]: Could not find module `reagent.gym.agents.agent`.
from reagent.gym.agents.agent import Agent

# pyre-fixme[21]: Could not find module `reagent.gym.envs`.
from reagent.gym.envs import Gym

# pyre-fixme[21]: Could not find module `reagent.gym.policies.predictor_policies`.
from reagent.gym.policies.predictor_policies import create_predictor_policy_from_model

# pyre-fixme[21]: Could not find module `reagent.gym.policies.random_policies`.
from reagent.gym.policies.random_policies import make_random_policy_for_env

# pyre-fixme[21]: Could not find module `reagent.gym.runners.gymrunner`.
from reagent.gym.runners.gymrunner import evaluate_for_n_episodes

# pyre-fixme[21]: Could not find module `reagent.gym.utils`.
from reagent.gym.utils import fill_replay_buffer

# pyre-fixme[21]: Could not find module `reagent.model_managers.union`.
from reagent.model_managers.union import ModelManager__Union

# pyre-fixme[21]: Could not find module `reagent.publishers.union`.
from reagent.publishers.union import FileSystemPublisher, ModelPublisher__Union

# pyre-fixme[21]: Could not find module `reagent.replay_memory.circular_replay_buffer`.
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer

# pyre-fixme[21]: Could not find module `reagent.replay_memory.utils`.
from reagent.replay_memory.utils import replay_buffer_to_pre_timeline_df

from .types import TableSpec


logger = logging.getLogger(__name__)


def initialize_seed(seed: int, env):
    pl.seed_everything(seed)
    env.seed(seed)
    env.action_space.seed(seed)


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
    # pyre-fixme[16]: Module `reagent` has no attribute `gym`.
    env = Gym(env_name=env_name)
    # pyre-fixme[16]: Module `reagent` has no attribute `gym`.
    random_policy = make_random_policy_for_env(env)
    # pyre-fixme[16]: Module `reagent` has no attribute `gym`.
    agent = Agent.create_for_env(env, policy=random_policy)
    return _offline_gym(env, agent, pkl_path, num_train_transitions, max_steps, seed)


def offline_gym_predictor(
    env_name: str,
    # pyre-fixme[11]: Annotation `ModelManager__Union` is not defined as a type.
    model: ModelManager__Union,
    # pyre-fixme[11]: Annotation `ModelPublisher__Union` is not defined as a type.
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
    # pyre-fixme[16]: Module `reagent` has no attribute `gym`.
    env = Gym(env_name=env_name)
    agent = make_agent_from_model(env, model, publisher, module_name)
    return _offline_gym(env, agent, pkl_path, num_train_transitions, max_steps, seed)


def _offline_gym(
    # pyre-fixme[11]: Annotation `Gym` is not defined as a type.
    env: Gym,
    # pyre-fixme[11]: Annotation `Agent` is not defined as a type.
    agent: Agent,
    pkl_path: str,
    num_train_transitions: int,
    max_steps: Optional[int],
    seed: int = 1,
):
    initialize_seed(seed, env)

    # pyre-fixme[16]: Module `reagent` has no attribute `replay_memory`.
    replay_buffer = ReplayBuffer(replay_capacity=num_train_transitions, batch_size=1)
    # pyre-fixme[16]: Module `reagent` has no attribute `gym`.
    fill_replay_buffer(env, replay_buffer, num_train_transitions, agent)
    if isinstance(env.action_space, gym.spaces.Discrete):
        is_discrete_action = True
    else:
        assert isinstance(env.action_space, gym.spaces.Box)
        is_discrete_action = False
    # pyre-fixme[16]: Module `reagent` has no attribute `replay_memory`.
    df = replay_buffer_to_pre_timeline_df(is_discrete_action, replay_buffer)
    logger.info(f"Saving dataset with {len(df)} samples to {pkl_path}")
    df.to_pickle(pkl_path)


PRE_TIMELINE_SUFFIX = "_pre_timeline_operator"


def timeline_operator(pkl_path: str, input_table_spec: TableSpec):
    """Loads a pandas parquet, converts to pyspark, and uploads df to Hive.
    Then call the timeline operator.
    """

    pd_df = pd.read_pickle(pkl_path)
    # pyre-fixme[16]: Module `reagent` has no attribute `data`.
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
    # pyre-fixme[16]: Module `reagent` has no attribute `data`.
    call_spark_class(spark, class_name="Timeline", args=json.dumps(arg))


def make_agent_from_model(
    env: Gym,
    model: ModelManager__Union,
    publisher: ModelPublisher__Union,
    module_name: str,
):
    publisher_manager = publisher.value
    assert isinstance(
        publisher_manager,
        # pyre-fixme[16]: Module `reagent` has no attribute `publishers`.
        FileSystemPublisher,
    ), f"publishing manager is type {type(publisher_manager)}, not FileSystemPublisher"
    module_names = model.value.serving_module_names()
    assert module_name in module_names, f"{module_name} not in {module_names}"
    torchscript_path = publisher_manager.get_latest_published_model(
        model.value, module_name
    )
    # pyre-fixme[16]: Module `torch` has no attribute `jit`.
    jit_model = torch.jit.load(torchscript_path)
    # pyre-fixme[16]: Module `reagent` has no attribute `gym`.
    policy = create_predictor_policy_from_model(jit_model)
    # pyre-fixme[16]: Module `reagent` has no attribute `gym`.
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
    # pyre-fixme[16]: Module `reagent` has no attribute `gym`.
    env = Gym(env_name=env_name)
    initialize_seed(1, env)
    agent = make_agent_from_model(env, model, publisher, module_name)

    # pyre-fixme[16]: Module `reagent` has no attribute `gym`.
    rewards = evaluate_for_n_episodes(
        n=num_eval_episodes, env=env, agent=agent, max_steps=max_steps
    )
    avg_reward = np.mean(rewards)
    logger.info(
        f"Average reward over {num_eval_episodes} is {avg_reward}.\n"
        f"List of rewards: {rewards}\n"
        f"Passing score bar: {passing_score_bar}"
    )
    assert avg_reward >= passing_score_bar, (
        f"{avg_reward} fails to pass the bar of {passing_score_bar}!"
    )
    return
