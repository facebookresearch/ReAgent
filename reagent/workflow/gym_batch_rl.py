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
from reagent.gym.agents.agent import Agent
from reagent.gym.envs.env_factory import EnvFactory
from reagent.gym.runners.gymrunner import evaluate_for_n_episodes
from reagent.gym.utils import fill_replay_buffer
from reagent.prediction.dqn_torch_predictor import (
    ActorTorchPredictor,
    DiscreteDqnTorchPredictor,
)
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.replay_memory.utils import replay_buffer_to_pre_timeline_df
from reagent.workflow.model_managers.union import ModelManager__Union
from reagent.workflow.predictor_policies import (
    ActorTorchPredictorPolicy,
    DiscreteDqnTorchPredictorPolicy,
)
from reagent.workflow.publishers.union import ModelPublisher__Union
from reagent.workflow.spark_utils import call_spark_class, get_spark_session
from reagent.workflow.training import identify_and_train_network
from reagent.workflow.types import RewardOptions, TableSpec
from reagent.workflow.validators.union import ModelValidator__Union


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
    env = EnvFactory.make(env_name)

    replay_buffer = ReplayBuffer.create_from_env(
        env=env, replay_memory_size=num_train_transitions, batch_size=1
    )
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


def create_predictor_policy_from_model(env: gym.Env, model: torch.nn.Module, **kwargs):
    if isinstance(env.action_space, gym.spaces.Discrete):
        assert "eval_temperature" in kwargs
        predictor = DiscreteDqnTorchPredictor(model)
        predictor.softmax_temperature = kwargs["eval_temperature"]
        return DiscreteDqnTorchPredictorPolicy(predictor)
    elif isinstance(env.action_space, gym.spaces.Box):
        assert len(env.action_space.shape) == 1
        predictor = ActorTorchPredictor(
            model, action_feature_ids=list(range(env.action_space.shape[0]))
        )
        return ActorTorchPredictorPolicy(predictor)
    else:
        raise NotImplementedError(f"{env.action_space} not supported")


def train_and_evaluate_gym(
    env_name: str,
    eval_temperature: float,
    num_eval_episodes: int,
    passing_score_bar: float,
    input_table_spec: TableSpec,
    model: ModelManager__Union,
    num_train_epochs: int,
    use_gpu: Optional[bool] = None,
    reward_options: Optional[RewardOptions] = None,
    warmstart_path: Optional[str] = None,
    validator: Optional[ModelValidator__Union] = None,
    publisher: Optional[ModelPublisher__Union] = None,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
):
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    initialize_seed(seed)
    training_output = identify_and_train_network(
        input_table_spec=input_table_spec,
        model=model,
        num_epochs=num_train_epochs,
        use_gpu=use_gpu,
        reward_options=reward_options,
        warmstart_path=warmstart_path,
        validator=validator,
        publisher=publisher,
    )

    env = EnvFactory.make(env_name)
    jit_model = torch.jit.load(training_output.output_path)
    policy = create_predictor_policy_from_model(
        env, jit_model, eval_temperature=eval_temperature
    )
    # since we already return softmax action, override action_extractor
    agent = Agent.create_for_env(
        env, policy=policy, action_extractor=policy.get_action_extractor()
    )
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
