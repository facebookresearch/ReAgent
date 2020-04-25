#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import json
import logging
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
from reagent.gym.agents.agent import Agent
from reagent.gym.agents.post_step import log_data_post_step
from reagent.gym.envs.env_factory import EnvFactory
from reagent.gym.policies import DiscreteRandomPolicy, Policy
from reagent.gym.runners.gymrunner import run_episode
from reagent.prediction.dqn_torch_predictor import DiscreteDqnTorchPredictor
from reagent.training.rl_dataset import RLDataset
from reagent.workflow.model_managers.union import ModelManager__Union
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

    policy = DiscreteRandomPolicy.create_for_env(env)
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


def upload_to_hive(pkl_path: str, input_table_spec: TableSpec):
    """ Loads a pandas parquet, converts to pyspark, and uploads df to Hive. """
    pd_df = pd.read_pickle(pkl_path)
    spark = get_spark_session()
    df = spark.createDataFrame(pd_df)
    tbl_name = f"{input_table_spec.table_name}{PRE_TIMELINE_SUFFIX}"
    df.write.mode("overwrite").saveAsTable(tbl_name)


def timeline_operator(input_table_spec: TableSpec):
    """ Call the timeline operator. """
    input_name = f"{input_table_spec.table_name}{PRE_TIMELINE_SUFFIX}"
    output_name = input_table_spec.table_name
    arg = {
        "startDs": "2019-01-01",
        "endDs": "2019-01-01",
        "addTerminalStateRow": True,
        "inputTableName": input_name,
        "outputTableName": output_name,
        "includePossibleActions": True,
        "percentileFunction": "percentile_approx",
        "rewardColumns": ["reward", "metrics"],
        "extraFeatureColumns": [],
    }
    input_json = json.dumps(arg)
    spark = get_spark_session()
    spark._jvm.com.facebook.spark.rl.Timeline.main(input_json)


class TorchPredictorPolicy(Policy):
    def __init__(self, predictor):
        self.predictor = predictor

    def act(self, state: np.ndarray, possible_actions_mask) -> int:
        state = torch.tensor(state).unsqueeze(0).to(torch.float32)
        return torch.tensor(self.predictor.policy(state).softmax)


def evaluate_gym(
    env: str,
    model,
    eval_temperature: float,
    num_eval_episodes: int,
    passing_score_bar: float,
    max_steps: Optional[int] = None,
):
    predictor = DiscreteDqnTorchPredictor(model)
    predictor.softmax_temperature = eval_temperature

    env = EnvFactory.make(env)
    policy = TorchPredictorPolicy(predictor)
    agent = Agent(policy=policy, action_extractor=lambda x: x.item())

    rewards = []
    for _ in range(num_eval_episodes):
        ep_reward = run_episode(env=env, agent=agent, max_steps=max_steps)
        rewards.append(ep_reward)

    avg_reward = np.mean(rewards)
    logger.info(
        f"Average reward over {num_eval_episodes} is {avg_reward}, "
        f"which passes the bar of {passing_score_bar}!\n"
        f"List of rewards: {rewards}"
    )
    assert avg_reward >= passing_score_bar


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
