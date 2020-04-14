#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
Policy Preprocessors are meant to transform gym.Env's observations for input into
the policy, more specifically the scorer. For example, gym.Env usually outputs numpy
arrays, while our scorer would expect Tensors (potentially GPU).
"""

import numpy as np
import torch
from reagent import types as rlt
from reagent.gym.types import PolicyPreprocessor


def numpy_policy_preprocessor(device: str = "cpu") -> PolicyPreprocessor:
    device = torch.device(device)

    def preprocess_obs(obs: np.array) -> rlt.PreprocessedState:
        return rlt.PreprocessedState.from_tensor(torch.tensor(obs).float().to(device))

    return preprocess_obs


def tiled_numpy_policy_preprocessor(
    num_actions: int, device: str = "cpu"
) -> PolicyPreprocessor:
    device = torch.device(device)

    def preprocess_obs(obs: np.array) -> rlt.PreprocessedStateAction:
        obs = torch.tensor(obs).float().to(device)
        tiled_state = torch.repeat_interleave(
            obs.unsqueeze(0), repeats=num_actions, axis=0
        )
        actions = torch.eye(num_actions)
        ts_size = tiled_state.size(0)
        a_size = actions.size(0)
        assert (
            ts_size == a_size
        ), f"state batch size {ts_size}, action batch size {a_size}"
        return rlt.PreprocessedStateAction.from_tensors(tiled_state, actions)

    return preprocess_obs
