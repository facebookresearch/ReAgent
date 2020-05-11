#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Any, Callable, List

import gym
import numpy as np
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer


logger = logging.getLogger(__name__)


try:
    from recsim.simulator.recsim_gym import RecSimGymEnv

    HAS_RECSIM = True
except ImportError:
    HAS_RECSIM = False
    logger.warning(f"ReplayBuffer.create_from_env() will not recognize RecSim env")


# Arguments: replay_buffer, obs, action, reward, terminal, log_prob
ReplayBufferInserter = Callable[[ReplayBuffer, Any, Any, float, bool, float], None]


def make_replay_buffer_inserter(env: gym.Env) -> ReplayBufferInserter:
    if HAS_RECSIM and isinstance(env, RecSimGymEnv):
        return RecSimReplayBufferInserter.create_for_env(env)
    return BasicReplayBufferInserter()


class BasicReplayBufferInserter:
    def __call__(
        self,
        replay_buffer: ReplayBuffer,
        obs: Any,
        action: Any,
        reward: float,
        terminal: bool,
        log_prob: float,
    ):
        replay_buffer.add(obs, action, reward, terminal, log_prob=log_prob)


class RecSimReplayBufferInserter:
    def __init__(
        self,
        *,
        num_docs: int,
        discrete_keys: List[str],
        box_keys: List[str],
        response_discrete_keys: List[str],
        response_box_keys: List[str],
    ):
        self.num_docs = num_docs
        self.discrete_keys = discrete_keys
        self.box_keys = box_keys
        self.response_discrete_keys = response_discrete_keys
        self.response_box_keys = response_box_keys

    @classmethod
    def create_for_env(cls, env: "RecSimGymEnv"):
        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Dict)
        user_obs_space = obs_space["user"]
        if not isinstance(user_obs_space, gym.spaces.Box):
            raise NotImplementedError(
                f"User observation space {type(user_obs_space)} is not supported"
            )

        doc_obs_space = obs_space["doc"]
        if not isinstance(doc_obs_space, gym.spaces.Dict):
            raise NotImplementedError(
                f"Doc space {type(doc_obs_space)} is not supported"
            )

        # Assume that all docs are in the same space

        discrete_keys: List[str] = []
        box_keys: List[str] = []

        doc_0_space = doc_obs_space["0"]

        if isinstance(doc_0_space, gym.spaces.Dict):
            for k, v in doc_obs_space["0"].spaces.items():
                if isinstance(v, gym.spaces.Discrete):
                    if v.n > 0:
                        discrete_keys.append(k)
                elif isinstance(v, gym.spaces.Box):
                    shape_dim = len(v.shape)
                    if shape_dim <= 1:
                        box_keys.append(k)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError(
                        f"Doc feature {k} with the observation space of {type(v)}"
                        " is not supported"
                    )
        elif isinstance(doc_0_space, gym.spaces.Box):
            pass
        else:
            raise NotImplementedError(f"Unknown space {doc_0_space}")

        response_space = obs_space["response"][0]
        assert isinstance(response_space, gym.spaces.Dict)
        response_box_keys: List[str] = []
        response_discrete_keys: List[str] = []
        for k, v in response_space.spaces.items():
            if isinstance(v, gym.spaces.Discrete):
                response_discrete_keys.append(k)
            elif isinstance(v, gym.spaces.Box):
                response_box_keys.append(k)
            else:
                raise NotImplementedError

        return cls(
            num_docs=len(doc_obs_space.spaces),
            discrete_keys=discrete_keys,
            box_keys=box_keys,
            response_box_keys=response_box_keys,
            response_discrete_keys=response_discrete_keys,
        )

    def __call__(
        self,
        replay_buffer: ReplayBuffer,
        obs: Any,
        action: Any,
        reward: float,
        terminal: bool,
        log_prob: float,
    ):
        user = obs["user"]

        kwargs = {}

        if self.box_keys or self.discrete_keys:
            doc_obs = obs["doc"]
            for k in self.box_keys:
                kwargs["doc_{k}"] = np.vstack([v[k] for v in doc_obs.values()])
            for k in self.discrete_keys:
                kwargs["doc_{k}"] = np.array([v[k] for v in doc_obs.values()])
        else:
            kwargs["doc"] = obs["doc"]

        # Responses

        for k in self.response_box_keys:
            kwargs["response_{k}"] = np.vstack([v[k] for v in obs["response"]])
        for k in self.response_discrete_keys:
            kwargs["response_{k}"] = np.arrray([v[k] for v in obs["response"]])

        replay_buffer.add(
            observation=user,
            action=action,
            reward=reward,
            terminal=terminal,
            log_prob=log_prob,
            **kwargs,
        )
