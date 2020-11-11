#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from collections import namedtuple
from typing import List, Optional

import gym
import numpy as np
from gym.utils import seeding
from gym.wrappers.time_limit import TimeLimit
from reagent.core.dataclasses import dataclass
from reagent.gym.envs.env_wrapper import EnvWrapper
from reagent.gym.envs.recsim import RecsimObsPreprocessor
from reagent.gym.envs.wrappers.recsim import ValueWrapper
from scipy.special import expit, logit


Document = namedtuple("Document", ["tap", "quality", "abandon"])


def simulate_reward(
    slate: List[Document], prng: np.random.RandomState  # pyre-ignore[11]
):
    reward = 0
    position = 0
    n = len(slate)
    if not n:
        return 0  # Bail if slate is empty
    comparison = slate[position].tap
    roll = prng.rand()
    done = comparison < roll
    while not done:
        reward += slate[position].quality
        comparison = 1 - slate[position].abandon
        roll = prng.rand()
        position += 1
        done = (comparison < roll) or (position >= n)
    return reward


def random_document(prng):
    p, q, r = prng.rand(), prng.rand(), prng.rand()
    return Document(expit(logit(p) + 1), q, expit(logit(r) - 2))


class ToyVMEnv(gym.Env):
    def __init__(self, slate_size: int):
        self.slate_size = slate_size
        self.action_space = gym.spaces.MultiDiscrete(
            [self.slate_size] * self.slate_size
        )
        self.observation_space = gym.spaces.Dict(
            {
                "user": gym.spaces.Box(low=0, high=1, shape=(1,)),
                "doc": gym.spaces.Dict(
                    {
                        str(k): gym.spaces.Box(
                            low=0, high=1, shape=(self.slate_size, 3)
                        )
                        for k in range(self.slate_size)
                    }
                ),
            }
        )
        self.response_space = gym.spaces.Dict({})
        self._doc_sampler = np.random.RandomState()
        self._reward_prng = np.random.RandomState()

    def seed(self, seed: Optional[int] = None):
        self._doc_sampler, seed1 = seeding.np_random(seed)
        _seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        self._reward_prng, seed2 = seeding.np_random(_seed2)
        return [seed1, seed2]

    def _sample_candidates(self):
        self.candidates = [
            random_document(self._doc_sampler) for _ in range(self.slate_size)
        ]
        n = len(self.candidates)
        return {
            "user": np.zeros((1,)),
            "doc": {
                str(k): np.array(self.candidates[k], dtype=np.float32) for k in range(n)
            },
        }

    def step(self, action):
        slate = [self.candidates[i] for i in action]
        reward = simulate_reward(slate, self._reward_prng)
        obs = self._sample_candidates()
        done = False
        info = {"documents": self.candidates}
        return obs, reward, done, info

    def reset(self):
        return self._sample_candidates()


def zero_augment(user, doc):
    return 0.0


@dataclass
class ToyVM(EnvWrapper):
    slate_size: int = 5
    max_episode_steps: int = 100
    initial_seed: Optional[int] = None

    def make(self):
        env = ValueWrapper(
            TimeLimit(
                ToyVMEnv(self.slate_size),
                max_episode_steps=self.max_episode_steps,
            ),
            zero_augment,
        )
        if self.initial_seed:
            env.seed(self.initial_seed)
        return env

    def action_extractor(self, actor_output):
        # Extract action from actor output
        return actor_output.action.squeeze()

    def obs_preprocessor(self, obs):
        preprocessor = RecsimObsPreprocessor.create_from_env(self)
        return preprocessor(obs)

    def serving_obs_preprocessor(self, obs):
        preprocessor = RecsimObsPreprocessor.create_from_env(self)
        return preprocessor(obs)
