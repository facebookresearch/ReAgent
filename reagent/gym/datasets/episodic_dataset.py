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
