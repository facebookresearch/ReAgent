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
from reagent.gym.agents.post_step import log_data_post_step
from reagent.gym.envs.env_factory import EnvFactory
from reagent.gym.policies.random_policies import make_random_policy_for_env
from reagent.gym.runners.gymrunner import run_episode
from reagent.prediction.dqn_torch_predictor import (
    ActorTorchPredictor,
    DiscreteDqnTorchPredictor,
)
from reagent.training.rl_dataset import RLDataset
from reagent.workflow.model_managers.union import ModelManager__Union
from reagent.workflow.predictor_policies import (
    ActorTorchPredictorPolicy,
    DiscreteDqnTorchPredictorPolicy,
)
from reagent.workflow.publishers.union import ModelPublisher__Union
from reagent.workflow.spark_utils import get_spark_session
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
    env: str,
    pkl_path: str,
    num_episodes_for_data_batch: int,
    max_steps: Optional[int],
    seed: Optional[int] = None,
):
    """
    Generate samples from a DiscreteRandomPolicy on the Gym environment and
    saves results in a pandas df parquet.
    """
    initialize_seed(seed)
    env = EnvFactory.make(env)
    policy = make_random_policy_for_env(env)

    dataset = RLDataset()
    for i in range(num_episodes_for_data_batch):
        logger.info(f"Starting episode {i}")
        post_step = log_data_post_step(dataset=dataset, mdp_id=str(i), env=env)
        agent = Agent.create_for_env(env, policy, post_transition_callback=post_step)
        run_episode(env=env, agent=agent, max_steps=max_steps)

    logger.info(f"Saving dataset with {len(dataset)} samples to {pkl_path}")
    df = dataset.to_pandas_df()
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
    input_json = json.dumps(arg)
    spark._jvm.com.facebook.spark.rl.Timeline.main(input_json)


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


def evaluate_gym(
    env: str,
    model: torch.nn.Module,
    eval_temperature: float,
    num_eval_episodes: int,
    passing_score_bar: float,
    max_steps: Optional[int] = None,
):
    env = EnvFactory.make(env)
    policy = create_predictor_policy_from_model(
        env, model, eval_temperature=eval_temperature
    )

    # since we already return softmax action, override action_extractor
    agent = Agent.create_for_env(
        env, policy=policy, action_extractor=policy.get_action_extractor()
    )

    rewards = []
    for _ in range(num_eval_episodes):
        ep_reward = run_episode(env=env, agent=agent, max_steps=max_steps)
        rewards.append(ep_reward)

    avg_reward = np.mean(rewards)
    logger.info(
        f"Average reward over {num_eval_episodes} is {avg_reward}.\n"
        f"List of rewards: {rewards}"
    )
    assert (
        avg_reward >= passing_score_bar
    ), f"{avg_reward} fails to pass the bar of {passing_score_bar}!"


def train_and_evaluate_gym(
    # for train
    input_table_spec: TableSpec,
    model: ModelManager__Union,
    num_train_epochs: int,
    use_gpu: bool = True,
    reward_options: Optional[RewardOptions] = None,
    warmstart_path: Optional[str] = None,
    validator: Optional[ModelValidator__Union] = None,
    publisher: Optional[ModelPublisher__Union] = None,
    seed: Optional[int] = None,
    # for eval
    env: str = None,
    eval_temperature: float = None,
    num_eval_episodes: int = None,
    passing_score_bar: float = None,
    max_steps: Optional[int] = None,
):
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

    jit_model = torch.jit.load(training_output.output_path)
    evaluate_gym(
        env=env,
        model=jit_model,
        eval_temperature=eval_temperature,
        num_eval_episodes=num_eval_episodes,
        passing_score_bar=passing_score_bar,
        max_steps=max_steps,
    )
    return
