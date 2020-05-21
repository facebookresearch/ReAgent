#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Any, Callable, List, Tuple

import gym
import numpy as np
from reagent.gym.types import Transition
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer


logger = logging.getLogger(__name__)


try:
    from recsim.simulator.recsim_gym import RecSimGymEnv

    HAS_RECSIM = True
except ImportError:
    HAS_RECSIM = False
    logger.warning(f"ReplayBuffer.create_from_env() will not recognize RecSim env")


# Arguments: replay_buffer, obs, action, reward, terminal, log_prob
ReplayBufferInserter = Callable[[ReplayBuffer, Transition], None]


def make_replay_buffer_inserter(env: gym.Env) -> ReplayBufferInserter:
    if HAS_RECSIM and isinstance(env.unwrapped, RecSimGymEnv):
        return RecSimReplayBufferInserter.create_for_env(env)
    return BasicReplayBufferInserter()


class BasicReplayBufferInserter:
    def __call__(self, replay_buffer: ReplayBuffer, transition: Transition):
        replay_buffer.add(**transition.asdict())


class RecSimReplayBufferInserter:
    def __init__(
        self,
        *,
        num_docs: int,
        num_responses: int,
        discrete_keys: List[str],
        box_keys: List[str],
        response_discrete_keys: List[Tuple[str, int]],
        response_box_keys: List[Tuple[str, Tuple[int]]],
        augmentation_discrete_keys: List[str],
        augmentation_box_keys: List[str],
    ):
        self.num_docs = num_docs
        self.num_responses = num_responses
        self.discrete_keys = discrete_keys
        self.box_keys = box_keys
        self.response_discrete_keys = response_discrete_keys
        self.response_box_keys = response_box_keys
        self.augmentation_discrete_keys = augmentation_discrete_keys
        self.augmentation_box_keys = augmentation_box_keys

    @classmethod
    def create_for_env(cls, env: gym.Env):
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
            for k, v in doc_0_space.spaces.items():
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

        augmentation_discrete_keys: List[str] = []
        augmentation_box_keys: List[str] = []
        augmentation = obs_space.spaces.get("augmentation", None)
        if augmentation is not None:
            aug_0_space = list(augmentation.spaces.values())[0]
            for k, v in aug_0_space.spaces.items():
                if isinstance(v, gym.spaces.Discrete):
                    if v.n > 0:
                        augmentation_discrete_keys.append(k)
                elif isinstance(v, gym.spaces.Box):
                    shape_dim = len(v.shape)
                    if shape_dim <= 1:
                        augmentation_box_keys.append(k)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError(
                        f"Augmentation {k} with the observation space "
                        f" of {type(v)} is not supported"
                    )

        response_space = obs_space["response"][0]
        assert isinstance(response_space, gym.spaces.Dict)
        response_box_keys: List[Tuple[str, Tuple[int]]] = []
        response_discrete_keys: List[Tuple[str, int]] = []
        for k, v in response_space.spaces.items():
            if isinstance(v, gym.spaces.Discrete):
                response_discrete_keys.append((k, v.n))
            elif isinstance(v, gym.spaces.Box):
                response_box_keys.append((k, v.shape))
            else:
                raise NotImplementedError

        return cls(
            num_docs=len(doc_obs_space.spaces),
            num_responses=len(obs_space["response"]),
            discrete_keys=discrete_keys,
            box_keys=box_keys,
            response_box_keys=response_box_keys,
            response_discrete_keys=response_discrete_keys,
            augmentation_box_keys=augmentation_box_keys,
            augmentation_discrete_keys=augmentation_discrete_keys,
        )

    def __call__(self, replay_buffer: ReplayBuffer, transition: Transition):
        transition_dict = transition.asdict()
        obs = transition_dict.pop("observation")
        user = obs["user"]

        kwargs = {}

        if self.box_keys or self.discrete_keys:
            doc_obs = obs["doc"]
            for k in self.box_keys:
                kwargs[f"doc_{k}"] = np.stack([v[k] for v in doc_obs.values()])
            for k in self.discrete_keys:
                kwargs[f"doc_{k}"] = np.array([v[k] for v in doc_obs.values()])
        else:
            kwargs["doc"] = np.stack(list(obs["doc"].values()))

        # Augmentation

        if self.augmentation_box_keys or self.augmentation_discrete_keys:
            aug_obs = obs["augmentation"]
            for k in self.augmentation_box_keys:
                kwargs[f"augmentation_{k}"] = np.stack([v[k] for v in aug_obs.values()])
            for k in self.augmentation_discrete_keys:
                kwargs[f"augmentation_{k}"] = np.array([v[k] for v in aug_obs.values()])

        # Responses

        response = obs["response"]
        # We need to handle None below because the first state won't have response
        for k, d in self.response_box_keys:
            if response is not None:
                kwargs[f"response_{k}"] = np.stack([v[k] for v in response])
            else:
                kwargs[f"response_{k}"] = np.zeros((self.num_responses, *d))
        for k, _n in self.response_discrete_keys:
            if response is not None:
                kwargs[f"response_{k}"] = np.array([v[k] for v in response])
            else:
                kwargs[f"response_{k}"] = np.zeros((self.num_responses,))

        transition_dict.update(kwargs)
        replay_buffer.add(observation=user, **transition_dict)
