#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Callable, Optional

import torch
from reagent.gym.agents.agent import Agent
from reagent.gym.envs import EnvWrapper
from reagent.gym.preprocessors import (
    make_replay_buffer_inserter,
    make_replay_buffer_trainer_preprocessor,
)
from reagent.gym.types import Trajectory, Transition
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class ReplayBufferDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        env: EnvWrapper,
        agent: Agent,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        training_frequency: int = 1,
        num_episodes: Optional[int] = None,
        max_steps: Optional[int] = None,
        post_episode_callback: Optional[Callable] = None,
        trainer_preprocessor=None,
        replay_buffer_inserter=None,
    ):
        super().__init__()
        self._env = env
        self._agent = agent
        self._replay_buffer = replay_buffer
        self._batch_size = batch_size
        self._training_frequency = training_frequency
        self._num_episodes = num_episodes
        self._max_steps = max_steps
        self._post_episode_callback = post_episode_callback
        self._trainer_preprocessor = trainer_preprocessor
        assert replay_buffer_inserter is not None
        self._replay_buffer_inserter = replay_buffer_inserter

    # TODO: Just use kwargs here?
    @classmethod
    def create_for_trainer(
        cls,
        trainer,
        env: EnvWrapper,
        agent: Agent,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        training_frequency: int = 1,
        num_episodes: Optional[int] = None,
        max_steps: Optional[int] = None,
        post_episode_callback: Optional[Callable] = None,
        trainer_preprocessor=None,
        replay_buffer_inserter=None,
        device=None,
    ):
        device = device or torch.device("cpu")
        if trainer_preprocessor is None:
            trainer_preprocessor = make_replay_buffer_trainer_preprocessor(
                trainer, device, env
            )

        if replay_buffer_inserter is None:
            replay_buffer_inserter = make_replay_buffer_inserter(env)

        return cls(
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            training_frequency=training_frequency,
            num_episodes=num_episodes,
            max_steps=max_steps,
            post_episode_callback=post_episode_callback,
            trainer_preprocessor=trainer_preprocessor,
            replay_buffer_inserter=replay_buffer_inserter,
        )

    def __iter__(self):
        mdp_id = 0
        global_num_steps = 0
        rewards = []

        # TODO: We probably should put member vars into local vars to
        # reduce indirection, improving perf

        while self._num_episodes is None or mdp_id < self._num_episodes:
            obs = self._env.reset()
            possible_actions_mask = self._env.possible_actions_mask
            terminal = False
            num_steps = 0
            episode_reward_sum = 0
            trajectory = Trajectory()
            while not terminal:
                action, log_prob = self._agent.act(obs, possible_actions_mask)
                next_obs, reward, terminal, info = self._env.step(action)
                next_possible_actions_mask = self._env.possible_actions_mask
                if self._max_steps is not None and num_steps >= self._max_steps:
                    terminal = True

                # Only partially filled. Agent can fill in more fields.
                transition = Transition(
                    mdp_id=mdp_id,
                    sequence_number=num_steps,
                    observation=obs,
                    action=action,
                    reward=float(reward),
                    terminal=bool(terminal),
                    log_prob=log_prob,
                    possible_actions_mask=possible_actions_mask,
                )
                trajectory.add_transition(transition)
                self._replay_buffer_inserter(self._replay_buffer, transition)
                episode_reward_sum += reward
                if (
                    global_num_steps % self._training_frequency == 0
                    and self._replay_buffer.size >= self._batch_size
                ):
                    train_batch = self._replay_buffer.sample_transition_batch(
                        batch_size=self._batch_size
                    )
                    if self._trainer_preprocessor:
                        train_batch = self._trainer_preprocessor(train_batch)
                    yield train_batch

                obs = next_obs
                possible_actions_mask = next_possible_actions_mask
                num_steps += 1
                global_num_steps += 1
                if self._agent.post_step:
                    self._agent.post_step(transition)
            if self._post_episode_callback:
                self._post_episode_callback(trajectory, info)

            rewards.append(episode_reward_sum)
            mdp_id += 1
            logger.info(
                f"Training episode: {mdp_id}, total episode reward = {episode_reward_sum}"
            )

        logger.info(f"Episode rewards during training: {rewards}")


class OfflineReplayBufferDataset(torch.utils.data.IterableDataset):
    """
    Simply sampling from the replay buffer
    """

    def __init__(
        self,
        env: EnvWrapper,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        num_batches: int,
        trainer_preprocessor=None,
    ):
        super().__init__()
        self._env = env
        self._replay_buffer = replay_buffer
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._trainer_preprocessor = trainer_preprocessor

    # TODO: Just use kwargs here?
    @classmethod
    def create_for_trainer(
        cls,
        trainer,
        env: EnvWrapper,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        num_batches: int,
        trainer_preprocessor=None,
        device=None,
    ):
        device = device or torch.device("cpu")
        if trainer_preprocessor is None:
            trainer_preprocessor = make_replay_buffer_trainer_preprocessor(
                trainer, device, env
            )

        return cls(
            env=env,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            num_batches=num_batches,
            trainer_preprocessor=trainer_preprocessor,
        )

    def __iter__(self):
        for _ in range(self._num_batches):
            train_batch = self._replay_buffer.sample_transition_batch(
                batch_size=self._batch_size
            )
            if self._trainer_preprocessor:
                train_batch = self._trainer_preprocessor(train_batch)
            yield train_batch
