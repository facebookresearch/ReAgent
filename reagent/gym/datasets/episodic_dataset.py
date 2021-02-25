#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import math
from typing import Optional, Callable

import torch
from reagent.gym.agents.agent import Agent
from reagent.gym.envs.gym import Gym
from reagent.gym.runners.gymrunner import run_episode


logger = logging.getLogger(__name__)


class EpisodicDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        env: Gym,
        agent: Agent,
        num_episodes: int,
        seed: int = 0,
        max_steps: Optional[int] = None,
    ):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.seed = seed
        self.max_steps = max_steps

    def __iter__(self):
        self.env.reset()
        for i in range(self.num_episodes):
            trajectory = run_episode(
                self.env, self.agent, max_steps=self.max_steps, mdp_id=i
            )
            yield trajectory.to_dict()

    def __len__(self):
        return self.num_episodes


class EpisodicDatasetDataloader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: EpisodicDataset,
        num_episodes_between_updates: int = 1,
        batch_size: int = 1,
        num_epochs: int = 1,
        collate_fn: Callable = lambda x: x,
    ):
        self._dataset_kind = torch.utils.data._DatasetKind.Iterable
        self.num_workers = 0

        self.dataset = dataset
        self.num_episodes_between_updates = num_episodes_between_updates
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.collate_fn = collate_fn

    def __iter__(self):
        trajectories_buffer = []
        for counter, traj in enumerate(self.dataset):
            trajectories_buffer.append(traj)
            if (len(trajectories_buffer) == self.num_episodes_between_updates) or (
                counter == (len(self.dataset) - 1)
            ):
                for _ in range(self.num_epochs):
                    random_order = torch.randperm(len(trajectories_buffer))
                    for i in range(0, len(trajectories_buffer), self.batch_size):
                        idx = random_order[i : i + self.batch_size]
                        yield self.collate_fn([trajectories_buffer[k] for k in idx])
                trajectories_buffer = []

    def __len__(self):
        return (
            math.floor(len(self.dataset) / self.num_episodes_between_updates)
            * math.ceil(self.num_episodes_between_updates / self.batch_size)
            + math.ceil(
                len(self.dataset) % self.num_episodes_between_updates / self.batch_size
            )
        ) * self.num_epochs
