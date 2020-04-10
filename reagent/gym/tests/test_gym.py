#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
Environments that require short training and evaluation time (<=10min)
can be tested in this file.
"""
import json
import logging
import random
import unittest
from typing import Optional

import gym
import numpy as np
import torch
from ruamel.yaml import YAML
from reagent.gym.agents.agent import Agent
from reagent.gym.agents.post_step import replay_buffer_post_step
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.samplers.continuous_sampler import GaussianSampler
from reagent.gym.policies.samplers.discrete_sampler import SoftmaxActionSampler
from reagent.gym.policies.scorers.continuous_scorer import sac_scorer
from reagent.gym.policies.scorers.discrete_scorer import (
    discrete_dqn_scorer,
    parametric_dqn_scorer,
)
from reagent.gym.preprocessors.action_preprocessors.action_preprocessor import (
    continuous_action_preprocessor,
    discrete_action_preprocessor,
)
from reagent.gym.preprocessors.policy_preprocessors.policy_preprocessor import (
    numpy_policy_preprocessor,
    tiled_numpy_policy_preprocessor,
)
from reagent.gym.preprocessors.trainer_preprocessors.trainer_preprocessor import (
    discrete_dqn_trainer_preprocessor,
    parametric_dqn_trainer_preprocessor,
    sac_trainer_preprocessor,
)
from reagent.gym.runners.gymrunner import run_episode
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer, ReplayElement
from reagent.tensorboardX import SummaryWriterContext
from reagent.workflow.model_managers.union import ModelManager__Union
from reagent.training.sac_trainer import SACTrainerParameters
from reagent.parameters import (
    CEMParameters,
    ContinuousActionModelParameters,
    DiscreteActionModelParameters,
    EvaluationParameters,
    FeedForwardParameters,
    MDNRNNParameters,
    RainbowDQNParameters,
    RLParameters,
    TD3ModelParameters,
    TD3TrainingParameters,
    TrainingParameters,
)
from reagent.test.base.utils import only_continuous_normalizer
from reagent.core.configuration import make_config_class
from reagent.parameters import NormalizationData
from reagent.workflow.types import RewardOptions
from reagent.training.dqn_trainer import DQNTrainer
from reagent.training.parametric_dqn_trainer import ParametricDQNTrainer
from reagent.training.sac_trainer import SACTrainer
import logging

logger = logging.getLogger(__name__)

SEED = 0


def build_normalizer(env):
    return {
        "state": NormalizationData(
            dense_normalization_parameters=only_continuous_normalizer(
                list(range(env.observation_space.shape[0])),
                env.observation_space.low,
                env.observation_space.high,
            )
        )
    }


def run_test(
    env: str,
    model: ModelManager__Union,
    replay_memory_size: int,
    train_every_ts: int,
    train_after_ts: int,
    num_episodes: int,
    max_steps: Optional[int],
    last_score_bar: float,
):
    env = gym.make(env)
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

    # case on the different trainers
    if isinstance(trainer, DQNTrainer):
        action_shape = np.array(actions).shape
        action_type = np.int32
        sampler = SoftmaxActionSampler()
        scorer = discrete_dqn_scorer(trainer.q_network)
        action_preprocessor = discrete_action_preprocessor
        policy_preprocessor = numpy_policy_preprocessor()
        trainer_preprocessor = discrete_dqn_trainer_preprocessor(
            len(actions), normalization
        )
    elif isinstance(trainer, ParametricDQNTrainer):
        action_shape = np.array(actions).shape
        action_type = np.float32
        sampler = SoftmaxActionSampler()
        scorer = parametric_dqn_scorer(len(actions), trainer.q_network)
        action_preprocessor = discrete_action_preprocessor
        policy_preprocessor = tiled_numpy_policy_preprocessor(len(actions))
        trainer_preprocessor = parametric_dqn_trainer_preprocessor(
            len(actions), normalization
        )
    elif isinstance(trainer, SACTrainer):
        action_shape = (1,)
        action_type = np.float32
        sampler = GaussianSampler(
            trainer.actor_network,
            trainer.min_action_range_tensor_serving,
            trainer.max_action_range_tensor_serving,
            trainer.min_action_range_tensor_training,
            trainer.max_action_range_tensor_training,
        )
        scorer = sac_scorer(trainer.actor_network)
        action_preprocessor = continuous_action_preprocessor
        policy_preprocessor = numpy_policy_preprocessor()
        trainer_preprocessor = sac_trainer_preprocessor()
    else:
        logger.info(f"trainer has type {type(trainer)} not supported!")
        import pdb

        pdb.set_trace()

    replay_buffer = ReplayBuffer(
        observation_shape=(env.observation_space.shape[0],),
        stack_size=1,
        replay_capacity=replay_memory_size,
        batch_size=trainer.minibatch_size,
        observation_dtype=np.float32,
        action_shape=action_shape,
        action_dtype=action_type,
        reward_shape=(),
        reward_dtype=np.float32,
        extra_storage_types=[
            ReplayElement("possible_actions_mask", action_shape, action_type),
            ReplayElement("log_prob", (), np.float32),
        ],
    )
    post_step = replay_buffer_post_step(
        replay_buffer=replay_buffer,
        trainer=trainer,
        trainer_preprocessor=trainer_preprocessor,
        training_freq=train_every_ts,
        batch_size=trainer.minibatch_size,
        replay_burnin=train_after_ts,
    )

    def run_episode_help():
        policy = Policy(
            scorer=scorer, sampler=sampler, policy_preprocessor=policy_preprocessor
        )
        return run_episode(
            env=env,
            policy=policy,
            action_preprocessor=action_preprocessor,
            post_step=post_step,
            max_steps=max_steps,
        )

    reward_history = []
    for i in range(num_episodes):
        logger.info(f"running episode {i}")
        ep_reward = run_episode_help()
        reward_history.append(ep_reward)

    assert (
        reward_history[-1] >= last_score_bar
    ), f"reward after {len(reward_history)} episodes is {reward_history[-1]} < {last_score_bar}...\nFull reward history: {reward_history}"

    return reward_history


def run_from_config(path):
    yaml = YAML(typ="safe")
    with open(path, "r") as f:
        config_dict = yaml.load(f.read())

    @make_config_class(run_test)
    class ConfigClass:
        pass

    config = ConfigClass(**config_dict)
    return run_test(**config.asdict())


class TestGym(unittest.TestCase):
    def setUp(self):
        SummaryWriterContext._reset_globals()
        logging.basicConfig(level=logging.INFO)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        random.seed(SEED)

    def test_discrete_dqn_cartpole(self):
        reward_history = run_from_config(
            "reagent/workflow/sample_configs/cartpole_discrete_dqn_online.yaml"
        )
        logger.info(f"Discrete DQN passes, with reward_history={reward_history}.")

    @unittest.skip("Skipping since training takes more than 10 min.")
    def test_parametric_dqn_cartpole(self):
        raise NotImplementedError("TODO: make model manager for PDQN")

    @unittest.skip("Skipping since training takes more than 10 min.")
    def test_sac_pendulum(self):
        SAC_PENDULUM_JSON = "reagent/test/gym/sac_pendulum_v0.json"
        SAC_PENDULUM_NUM_EPISODES = 50
        PENDULUM_SCORE_BAR = -750
        raise NotImplementedError("TODO: make model manager for SAC")


if __name__ == "__main__":
    unittest.main()
