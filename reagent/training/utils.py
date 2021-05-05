#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import torch
import torch.nn.functional as F


EPS = np.finfo(float).eps.item()


def rescale_actions(
    actions: torch.Tensor,
    new_min: torch.Tensor,
    new_max: torch.Tensor,
    prev_min: torch.Tensor,
    prev_max: torch.Tensor,
) -> torch.Tensor:
    """Scale from [prev_min, prev_max] to [new_min, new_max]"""
    assert torch.all(prev_min <= actions) and torch.all(
        actions <= prev_max
    ), f"{actions} has values outside of [{prev_min}, {prev_max}]."
    assert torch.all(
        new_min <= new_max
    ), f"{new_min} is (has coordinate) greater than {new_max}."
    prev_range = prev_max - prev_min
    new_range = new_max - new_min
    return ((actions - prev_min) / prev_range) * new_range + new_min


def whiten(x: torch.Tensor, subtract_mean: bool) -> torch.Tensor:
    numer = x
    if subtract_mean:
        numer -= x.mean()
    return numer / (x.std() + EPS)


def discounted_returns(rewards: torch.Tensor, gamma: float = 0) -> torch.Tensor:
    """Perform rollout to compute reward to go
    and do a baseline subtraction."""
    if gamma == 0:
        return rewards.float()
    else:
        R = 0
        returns = []
        for r in rewards.numpy()[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        return torch.tensor(returns).float()


def gen_permutations(seq_len: int, num_action: int) -> torch.Tensor:
    """
    generate all seq_len permutations for a given action set
    the return shape is (SEQ_LEN, PERM_NUM, ACTION_DIM)
    """
    all_permut = torch.cartesian_prod(*[torch.arange(num_action)] * seq_len)
    if seq_len == 1:
        all_permut = all_permut.unsqueeze(1)
    all_permut = F.one_hot(all_permut, num_action).transpose(0, 1)
    return all_permut.float()
