#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List, Optional

import gym
import numpy as np
import torch
from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader, decimal_friendly_collate
from reagent.gym.agents.agent import Agent
from reagent.gym.envs.env_factory import EnvFactory
from reagent.gym.runners.gymrunner import run_episode
from reagent.preprocessing.batch_preprocessor import BatchPreprocessor
from reagent.training.rl_trainer_pytorch import RLTrainer
from reagent.workflow.spark_utils import get_spark_session
from reagent.workflow.types import Dataset, ReaderOptions
from reagent.workflow_utils.page_handler import (
    EvaluationPageHandler,
    TrainingPageHandler,
    feed_pages,
)


logger = logging.getLogger(__name__)


def get_table_row_count(parquet_url: str):
    spark = get_spark_session()
    return spark.read.parquet(parquet_url).count()


# TODO(kaiwenw): paralellize preprocessing by putting into transform of petastorm reader
def collate_and_preprocess(batch_preprocessor: BatchPreprocessor):
    def collate_fn(batch_list: List[Dict]):
        batch = decimal_friendly_collate(batch_list)
        return batch_preprocessor(batch)

    return collate_fn


# TMP for debugging
def evaluate_gym(env: str, model: torch.nn.Module, num_eval_episodes=5):
    from reagent.prediction.dqn_torch_predictor import (
        DiscreteDqnTorchPredictor,
        ActorTorchPredictor,
    )
    from reagent.workflow.predictor_policies import (
        DiscreteDqnTorchPredictorPolicy,
        ActorTorchPredictorPolicy,
    )

    env = EnvFactory.make(env)
    if isinstance(env.action_space, gym.spaces.Discrete):
        predictor = DiscreteDqnTorchPredictor(model)
        predictor.softmax_temperature = 0.01
        policy = DiscreteDqnTorchPredictorPolicy(predictor)
    elif isinstance(env.action_space, gym.spaces.Box):
        assert len(env.action_space.shape) == 1
        predictor = ActorTorchPredictor(
            model, action_feature_ids=list(range(env.action_space.shape[0]))
        )
        policy = ActorTorchPredictorPolicy(predictor)
    else:
        raise NotImplementedError(f"{env.action_space} not supported")

    # since we already return softmax action, override action_extractor
    agent = Agent.create_for_env(
        env, policy=policy, action_extractor=policy.get_action_extractor()
    )

    rewards = []
    for _ in range(num_eval_episodes):
        ep_reward = run_episode(env=env, agent=agent)
        rewards.append(ep_reward)

    avg_reward = np.mean(rewards)
    logger.info(
        f"Average reward over {num_eval_episodes} is {avg_reward}.\n"
        f"List of rewards: {rewards}"
    )


def train_and_evaluate_generic(
    model,  # tmp
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    trainer: RLTrainer,
    num_epochs: int,
    use_gpu: bool,
    batch_preprocessor: BatchPreprocessor,
    train_page_handler: TrainingPageHandler,
    eval_page_handler: EvaluationPageHandler,
    reader_options: Optional[ReaderOptions] = None,
):
    reader_options = reader_options or ReaderOptions()

    train_dataset_num_rows = get_table_row_count(train_dataset.parquet_url)
    eval_dataset_num_rows = None
    if eval_dataset is not None:
        eval_dataset_num_rows = get_table_row_count(eval_dataset.parquet_url)

    logger.info(
        f"train_data_num: {train_dataset_num_rows}, "
        f"eval_data_num: {eval_dataset_num_rows}"
    )

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch} start feeding training data")
        data_reader = make_batch_reader(
            train_dataset.parquet_url,
            num_epochs=1,
            reader_pool_type=reader_options.petastorm_reader_pool_type,
        )
        with DataLoader(
            data_reader,
            batch_size=trainer.minibatch_size,
            collate_fn=collate_and_preprocess(batch_preprocessor),
        ) as data_loader:
            feed_pages(
                data_loader,
                train_dataset_num_rows,
                epoch,
                trainer.minibatch_size,
                use_gpu,
                train_page_handler,
            )

        evaluate_gym("Pendulum-v0", model.build_serving_module())

        if not eval_dataset:
            continue

        logger.info(f"Epoch {epoch} start feeding evaluation data")
        eval_data_reader = make_batch_reader(
            eval_dataset.parquet_url,
            num_epochs=1,
            reader_pool_type=reader_options.petastorm_reader_pool_type,
        )
        with DataLoader(
            eval_data_reader,
            batch_size=trainer.minibatch_size,
            collate_fn=collate_and_preprocess(batch_preprocessor),
        ) as eval_data_loader:
            feed_pages(
                eval_data_loader,
                eval_dataset_num_rows,
                epoch,
                trainer.minibatch_size,
                use_gpu,
                eval_page_handler,
            )
