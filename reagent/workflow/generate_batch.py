#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
import gym
import pandas as pd
import pyspark
import pdb
import torch
import numpy as np
from typing import Optional

from reagent.workflow.model_managers.union import ModelManager__Union
from reagent.training.dqn_trainer import DQNTrainer
from reagent.gym.policies.policy import DiscreteRandomPolicy
from reagent.training.rl_dataset import RLDataset
from reagent.gym.preprocessors.action_preprocessors.action_preprocessor import (
    continuous_action_preprocessor,
    discrete_action_preprocessor,
)
from reagent.gym.agents.post_step import log_data_post_step
from reagent.workflow.types import RewardOptions
from reagent.gym.tests.test_gym import build_normalizer
from reagent.gym.runners.gymrunner import run_episode
from reagent.workflow.types import TableSpec
from reagent.workflow.spark_utils import get_spark_session

logger = logging.getLogger(__name__)


def offline_gym(
    env: str,
    pkl_path: str,
    model: ModelManager__Union,
    num_episodes: int,
    max_steps: int,
    seed: Optional[int] = None,
):
    """
    Generate samples from a DiscreteRandomPolicy on the Gym environment and
    saves results in a pandas df parquet.
    """
    env = gym.make(env)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)

    normalization = build_normalizer(env)
    logger.info(f"Normalization is {normalization}")

    manager = model.value
    actions = manager.trainer_param.actions
    logger.info(f"Actions are {actions}")

    manager.initialize_trainer(
        use_gpu=False,
        reward_options=RewardOptions(),
        normalization_data_map=normalization,
    )
    trainer = manager.trainer
    policy = DiscreteRandomPolicy(len(actions))

    # case on the different trainers
    if isinstance(trainer, DQNTrainer):
        action_preprocessor = discrete_action_preprocessor
    else:
        logger.info(f"trainer has type {type(trainer)} not supported!")
        pdb.set_trace()

    dataset = RLDataset()
    reward_history = []
    for i in range(num_episodes):
        logger.info(f"running episode {i}")
        post_step = log_data_post_step(dataset, action_preprocessor, str(i))
        ep_reward = run_episode(
            env=env,
            policy=policy,
            action_preprocessor=action_preprocessor,
            post_step=post_step,
            max_steps=max_steps,
        )
        reward_history.append(ep_reward)

    logger.info(f"Saving dataset with {len(dataset)} samples to {pkl_path}")
    df = dataset.to_pandas_df()
    df.to_pickle(pkl_path)


def upload_to_hive(pkl_path: str, input_table_spec: TableSpec):
    """ Loads a pandas parquet, converts to pyspark, and uploads df to Hive. """
    pd_df = pd.read_pickle(pkl_path)
    spark = get_spark_session()
    df = spark.createDataFrame(pd_df)
    tbl_name = f"{input_table_spec.table_name}_pre_timeline_operator"
    df.write.mode("overwrite").saveAsTable(tbl_name)


def rl_timeline_operator(input_table_spec: TableSpec):
    import os
    import json

    input_name = f"{input_table_spec.table_name}_pre_timeline_operator"
    output_name = input_table_spec.table_name
    eval_name = f"{input_table_spec.table_name}_eval"
    arg = {
        "timeline": {
            "startDs": "2019-01-01",
            "endDs": "2019-01-01",
            "addTerminalStateRow": True,
            "actionDiscrete": True,
            "inputTableName": input_name,
            "outputTableName": output_name,
            "evalTableName": eval_name,
            "numOutputShards": 1,
            "includePossibleActions": True,
            "percentileFunction": "percentile_approx",
            "rewardColumns": ["reward", "metrics"],
            "extraFeatureColumns": []
        },
        "query": {
            "tableSample": 100,
            "actions": ["0", "1"]
        },
    }
    # spark.sparkContext.addPyFile("preprocessing/target/rl-preprocessing-1.1.jar")
    input_json = json.dumps(arg).replace('\"', '\\\"')
    cmd = f"/usr/local/spark/bin/spark-submit --class com.facebook.spark.rl.Preprocessor preprocessing/target/rl-preprocessing-1.1.jar \"{input_json}\""
    print("Command is ", cmd)
    os.system(cmd)
    
    