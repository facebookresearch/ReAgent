#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
import os
import unittest
from typing import Dict, List

import gym
import numpy as np
import reagent.types as rlt
import torch
from reagent.evaluation.world_model_evaluator import (
    FeatureImportanceEvaluator,
    FeatureSensitivityEvaluator,
)
from reagent.gym.agents.agent import Agent
from reagent.gym.agents.post_step import add_replay_buffer_post_step
from reagent.gym.envs.env_factory import EnvFactory
from reagent.gym.envs.pomdp.state_embed_env import StateEmbedEnvironment
from reagent.gym.policies.random_policies import make_random_policy_for_env
from reagent.gym.preprocessors import make_replay_buffer_trainer_preprocessor
from reagent.gym.runners.gymrunner import run_episode
from reagent.gym.tests.utils import build_normalizer
from reagent.models.world_model import MemoryNetwork
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.test.base.horizon_test_base import HorizonTestBase
from reagent.training.world_model.mdnrnn_trainer import MDNRNNTrainer
from reagent.workflow.model_managers.union import ModelManager__Union
from reagent.workflow.types import RewardOptions
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

curr_dir = os.path.dirname(__file__)

SEED = 0


def print_mdnrnn_losses(epoch, batch_num, losses):
    logger.info(
        f"Printing loss for Epoch {epoch}, Batch {batch_num};\n"
        f"loss={losses['loss']}, bce={losses['bce']},"
        f"gmm={losses['gmm']}, mse={losses['mse']} \n"
    )


def create_rb(env, batch_size, seq_len, desired_size):
    rb = ReplayBuffer.create_from_env(
        env=env,
        replay_memory_size=desired_size,
        batch_size=batch_size,
        stack_size=seq_len,
        # we want sequence of actions too for WorldModel
        return_everything_as_stack=True,
    )
    random_policy = make_random_policy_for_env(env)
    post_step = add_replay_buffer_post_step(rb)
    agent = Agent.create_for_env(
        env, policy=random_policy, post_transition_callback=post_step
    )
    while not rb.is_full():
        run_episode(env, agent)
    return rb


def calculate_feature_importance(
    env: gym.Env,
    trainer: MDNRNNTrainer,
    use_gpu: bool,
    test_batch: rlt.PreprocessedMemoryNetworkInput,
):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert len(env.observation_space.shape) == 1
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    feature_importance_evaluator = FeatureImportanceEvaluator(
        trainer,
        discrete_action=True,
        state_feature_num=state_dim,
        action_feature_num=action_dim,
        sorted_state_feature_start_indices=list(range(state_dim)),
        sorted_action_feature_start_indices=list(range(action_dim)),
    )
    feature_loss_vector = feature_importance_evaluator.evaluate(test_batch)[
        "feature_loss_increase"
    ]
    feature_importance_map = {}
    for i in range(action_dim):
        print(
            "action {}, feature importance: {}".format(i, feature_loss_vector[i].item())
        )
        feature_importance_map[f"action{i}"] = feature_loss_vector[i].item()
    for i in range(state_dim):
        print(
            "state {}, feature importance: {}".format(
                i, feature_loss_vector[i + action_dim].item()
            )
        )
        feature_importance_map[f"state{i}"] = feature_loss_vector[i + action_dim].item()
    return feature_importance_map


def calculate_feature_sensitivity(
    env: gym.Env,
    trainer: MDNRNNTrainer,
    use_gpu: bool,
    test_batch: rlt.PreprocessedMemoryNetworkInput,
):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert len(env.observation_space.shape) == 1
    state_dim = env.observation_space.shape[0]
    feature_sensitivity_evaluator = FeatureSensitivityEvaluator(
        trainer,
        state_feature_num=state_dim,
        sorted_state_feature_start_indices=list(range(state_dim)),
    )
    feature_sensitivity_vector = feature_sensitivity_evaluator.evaluate(test_batch)[
        "feature_sensitivity"
    ]
    feature_sensitivity_map = {}
    for i in range(state_dim):
        feature_sensitivity_map["state" + str(i)] = feature_sensitivity_vector[i].item()
        print(
            "state {}, feature sensitivity: {}".format(
                i, feature_sensitivity_vector[i].item()
            )
        )
    return feature_sensitivity_map


def train_mdnrnn(
    env: gym.Env,
    trainer: MDNRNNTrainer,
    trainer_preprocessor,
    num_train_episodes: int,
    seq_len: int,
    batch_size: int,
    num_train_epochs: int,
    # for optional validation
    test_replay_buffer=None,
):
    train_replay_buffer = create_rb(
        env,
        batch_size,
        seq_len,
        # pyre-fixme[16]: `Env` has no attribute `_max_episode_steps`.
        num_train_episodes * env._max_episode_steps,
    )
    num_batch_per_epoch = train_replay_buffer.size // batch_size
    logger.info("Made RBs, starting to train now!")
    for epoch in range(num_train_epochs):
        for i in range(num_batch_per_epoch):
            batch = train_replay_buffer.sample_transition_batch_tensor(
                batch_size=batch_size
            )
            preprocessed_batch = trainer_preprocessor(batch)
            losses = trainer.train(preprocessed_batch)
            print_mdnrnn_losses(epoch, i, losses)

        # validation
        if test_replay_buffer is not None:
            with torch.no_grad():
                trainer.memory_network.mdnrnn.eval()
                test_batch = test_replay_buffer.sample_transition_batch_tensor(
                    batch_size=batch_size
                )
                preprocessed_test_batch = trainer_preprocessor(test_batch)
                valid_losses = trainer.get_loss(preprocessed_test_batch)
                print_mdnrnn_losses(epoch, "validation", valid_losses)
                trainer.memory_network.mdnrnn.train()
    return trainer


def train_mdnrnn_and_compute_feature_stats(
    env: str,
    model: ModelManager__Union,
    num_train_episodes: int,
    num_test_episodes: int,
    seq_len: int,
    batch_size: int,
    num_train_epochs: int,
    use_gpu: bool,
    # pyre-fixme[9]: saved_mdnrnn_path has type `str`; used as `None`.
    saved_mdnrnn_path: str = None,
):
    """ Train MDNRNN Memory Network and compute feature importance/sensitivity. """
    env: gym.Env = EnvFactory.make(env)
    env.seed(SEED)

    manager = model.value
    trainer = manager.initialize_trainer(
        use_gpu=use_gpu,
        reward_options=RewardOptions(),
        normalization_data_map=build_normalizer(env),
    )

    device = "cuda" if use_gpu else "cpu"
    trainer_preprocessor = make_replay_buffer_trainer_preprocessor(trainer, device)
    test_replay_buffer = create_rb(
        env,
        batch_size,
        seq_len,
        # pyre-fixme[16]: `Env` has no attribute `_max_episode_steps`.
        num_test_episodes * env._max_episode_steps,
    )

    if saved_mdnrnn_path is None:
        # train from scratch
        trainer = train_mdnrnn(
            env=env,
            trainer=trainer,
            trainer_preprocessor=trainer_preprocessor,
            num_train_episodes=num_train_episodes,
            seq_len=seq_len,
            batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            test_replay_buffer=test_replay_buffer,
        )
    else:
        # load a pretrained model, and just evaluate it
        trainer.memory_network.mdnrnn.load_state_dict(torch.load(saved_mdnrnn_path))

    with torch.no_grad():
        trainer.memory_network.mdnrnn.eval()
        test_batch = test_replay_buffer.sample_transition_batch_tensor(
            batch_size=test_replay_buffer.size
        )
        preprocessed_test_batch = trainer_preprocessor(test_batch)
        feature_importance = calculate_feature_importance(
            env=env,
            trainer=trainer,
            use_gpu=use_gpu,
            test_batch=preprocessed_test_batch,
        )

        feature_sensitivity = calculate_feature_sensitivity(
            env=env,
            trainer=trainer,
            use_gpu=use_gpu,
            test_batch=preprocessed_test_batch,
        )

        trainer.memory_network.mdnrnn.train()
    return feature_importance, feature_sensitivity


def create_embed_rl_dataset(
    env: gym.Env,
    memory_network: MemoryNetwork,
    num_state_embed_episodes: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    use_gpu: bool,
):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert len(env.observation_space.shape) == 1
    logger.info("Starting to create embedded RL Dataset!")

    # seqlen+1 because MDNRNN embeds the first seq_len steps and then
    # the embedded state will be concatenated with the last step
    # Ie.. (o1,o2,...,on) -> RNN -> h1,h2,...,hn
    # and we set s_{n+1} = [o_{n+1}, h_n]
    embed_env = StateEmbedEnvironment(
        gym_env=env, mdnrnn=memory_network, max_embed_seq_len=seq_len + 1
    )
    # now create a filled replay buffer of embeddings
    # new obs shape dim = state_dim + hidden_dim
    # pyre-fixme[16]: `Env` has no attribute `_max_episode_steps`.
    embed_rb_capacity = num_state_embed_episodes * env._max_episode_steps

    embed_rb = ReplayBuffer.create_from_env(
        env=embed_env,
        replay_memory_size=embed_rb_capacity,
        batch_size=batch_size,
        stack_size=1,
    )

    device = "cuda" if use_gpu else "cpu"
    policy = make_random_policy_for_env(embed_env)
    post_step = add_replay_buffer_post_step(embed_rb)
    agent = Agent.create_for_env(
        env=embed_env, policy=policy, post_transition_callback=post_step, device=device
    )

    # only an approximation, since episodes may not run to max steps
    with tqdm(total=num_state_embed_episodes, desc="approx filling embed_rb") as pbar:
        while not embed_rb.is_full():
            run_episode(env=embed_env, agent=agent)
            pbar.update()

    batch = embed_rb.sample_transition_batch_tensor(batch_size=embed_rb_capacity)
    state_min = min(batch.state.min(), batch.next_state.min()).item()
    state_max = max(batch.state.max(), batch.next_state.max()).item()
    logger.info(
        f"Finished making embed dataset with size {embed_rb.size}, "
        f"min {state_min}, max {state_max}"
    )
    return embed_rb, state_min, state_max


def train_mdnrnn_and_train_on_embedded_env(
    env: str,
    embedding_model: ModelManager__Union,
    num_embedding_train_episodes: int,
    seq_len: int,
    batch_size: int,
    num_embedding_train_epochs: int,
    train_model: ModelManager__Union,
    num_state_embed_episodes: int,
    num_agent_train_epochs: int,
    num_agent_eval_epochs: int,
    use_gpu: bool,
    passing_score_bar: float,
    # pyre-fixme[9]: saved_mdnrnn_path has type `str`; used as `None`.
    saved_mdnrnn_path: str = None,
):
    """ Train an agent on embedded states by the MDNRNN. """
    # pyre-fixme[9]: env has type `str`; used as `Env`.
    env = EnvFactory.make(env)
    # pyre-fixme[16]: `str` has no attribute `seed`.
    env.seed(SEED)

    embedding_manager = embedding_model.value
    embedding_trainer = embedding_manager.initialize_trainer(
        use_gpu=use_gpu,
        reward_options=RewardOptions(),
        normalization_data_map=build_normalizer(env),
    )

    device = "cuda" if use_gpu else "cpu"
    embedding_trainer_preprocessor = make_replay_buffer_trainer_preprocessor(
        embedding_trainer, device
    )
    if saved_mdnrnn_path is None:
        # train from scratch
        embedding_trainer = train_mdnrnn(
            # pyre-fixme[6]: Expected `Env` for 1st param but got `str`.
            env=env,
            trainer=embedding_trainer,
            trainer_preprocessor=embedding_trainer_preprocessor,
            num_train_episodes=num_embedding_train_episodes,
            seq_len=seq_len,
            batch_size=batch_size,
            num_train_epochs=num_embedding_train_epochs,
        )
    else:
        # load a pretrained model, and just evaluate it
        embedding_trainer.memory_network.mdnrnn.load_state_dict(
            torch.load(saved_mdnrnn_path)
        )

    # create embedding dataset
    embed_rb, state_min, state_max = create_embed_rl_dataset(
        # pyre-fixme[6]: Expected `Env` for 1st param but got `str`.
        env=env,
        memory_network=embedding_trainer.memory_network,
        num_state_embed_episodes=num_state_embed_episodes,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_dim=embedding_trainer.params.hidden_size,
        use_gpu=use_gpu,
    )
    embed_env = StateEmbedEnvironment(
        # pyre-fixme[6]: Expected `Env` for 1st param but got `str`.
        gym_env=env,
        mdnrnn=embedding_trainer.memory_network,
        max_embed_seq_len=seq_len,
        state_min_value=state_min,
        state_max_value=state_max,
    )
    agent_manager = train_model.value
    agent_trainer = agent_manager.initialize_trainer(
        use_gpu=use_gpu,
        reward_options=RewardOptions(),
        normalization_data_map=build_normalizer(embed_env),
    )
    device = "cuda" if use_gpu else "cpu"
    agent_trainer_preprocessor = make_replay_buffer_trainer_preprocessor(
        agent_trainer, device
    )
    num_batch_per_epoch = embed_rb.size // batch_size
    for epoch in range(num_agent_train_epochs):
        for _ in tqdm(range(num_batch_per_epoch), desc=f"epoch {epoch}"):
            batch = embed_rb.sample_transition_batch_tensor(batch_size=batch_size)
            preprocessed_batch = agent_trainer_preprocessor(batch)
            agent_trainer.train(preprocessed_batch)

    # evaluate model
    rewards = []
    policy = agent_manager.create_policy(serving=False)
    agent = Agent.create_for_env(embed_env, policy=policy, device=device)
    for i in range(num_agent_eval_epochs):
        ep_reward = run_episode(env=embed_env, agent=agent)
        rewards.append(ep_reward)
        logger.info(f"Finished eval episode {i} with reward {ep_reward}.")
    logger.info(f"Average eval reward is {np.mean(rewards)}.")
    assert (
        np.mean(rewards) >= passing_score_bar
    ), f"average reward doesn't pass our bar {passing_score_bar}"
    return rewards


class TestWorldModel(HorizonTestBase):
    @staticmethod
    def verify_result(result_dict: Dict[str, float], expected_top_features: List[str]):
        top_feature = max(result_dict, key=result_dict.get)
        assert (
            top_feature in expected_top_features
        ), f"top_feature: {top_feature}, expected_top_features: {expected_top_features}"

    def test_mdnrnn(self):
        """ Test MDNRNN feature importance and feature sensitivity. """
        config_path = "configs/world_model/cartpole_features.yaml"
        feature_importance, feature_sensitivity = self.run_from_config(
            run_test=train_mdnrnn_and_compute_feature_stats,
            config_path=os.path.join(curr_dir, config_path),
            use_gpu=False,
        )
        TestWorldModel.verify_result(feature_importance, ["state3"])
        TestWorldModel.verify_result(feature_sensitivity, ["state3"])
        logger.info("MDNRNN feature test passes!")

    def test_world_model(self):
        """ Train DQN on POMDP given features from world model. """
        config_path = "configs/world_model/discrete_dqn_string.yaml"
        HorizonTestBase.run_from_config(
            run_test=train_mdnrnn_and_train_on_embedded_env,
            config_path=os.path.join(curr_dir, config_path),
            use_gpu=False,
        )
        logger.info("World model test passes!")


if __name__ == "__main__":
    unittest.main()
