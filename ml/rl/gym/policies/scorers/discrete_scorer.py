#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import ml.rl.types as rlt
import torch
import torch.nn.functional as F
from ml.rl.gym.types import Scorer
from ml.rl.models.base import ModelBase


def discrete_dqn_scorer(q_network: ModelBase) -> Scorer:
    @torch.no_grad()
    def score(preprocessed_obs: rlt.PreprocessedState) -> torch.Tensor:
        q_network.eval()
        scores = q_network(preprocessed_obs).q_values
        q_network.train()
        return scores

    return score


def parametric_dqn_scorer(num_actions: int, q_network: ModelBase) -> Scorer:
    @torch.no_grad()
    def score(preprocessed_obs: rlt.PreprocessedStateAction) -> torch.Tensor:
        q_network.eval()
        scores = q_network(preprocessed_obs).q_value.view(-1)
        assert (
            scores.size(0) == num_actions
        ), f"scores size is {scores.size(0)}, num_actions is {num_actions}"
        q_network.train()
        return F.log_softmax(scores)

    return score
