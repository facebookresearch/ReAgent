#!/usr/bin/env python3

from typing import Optional, Union

import torch
from reagent.gym.agents.agent import Agent
from reagent.gym.envs import EnvWrapper
from reagent.gym.preprocessors import (
    make_replay_buffer_inserter,
    make_replay_buffer_trainer_preprocessor,
)
from reagent.gym.types import Transition
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer


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
        trainer_preprocessor=None,
        replay_buffer_inserter=None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)

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
            trainer_preprocessor=trainer_preprocessor,
            replay_buffer_inserter=replay_buffer_inserter,
        )

    def __iter__(self):
        mdp_id = 0
        global_num_steps = 0

        # TODO: We probably should put member vars into local vars to
        # reduce indirection, improving perf

        while self._num_episodes is None or mdp_id < self._num_episodes:
            obs = self._env.reset()
            possible_actions_mask = self._env.possible_actions_mask
            terminal = False
            num_steps = 0
            while not terminal:
                action, log_prob = self._agent.act(obs, possible_actions_mask)
                next_obs, reward, terminal, _ = self._env.step(action)
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
                self._replay_buffer_inserter(self._replay_buffer, transition)
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

            mdp_id += 1
