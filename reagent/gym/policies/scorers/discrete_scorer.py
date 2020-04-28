#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.gym.types import Scorer
from reagent.models.base import ModelBase


def discrete_dqn_scorer(q_network: ModelBase) -> Scorer:
    @torch.no_grad()
    def score(preprocessed_obs: rlt.PreprocessedState) -> torch.Tensor:
        q_network.eval()
        scores = q_network(preprocessed_obs).q_values
        q_network.train()
        return scores

    return score


def discrete_dqn_serving_scorer(q_network: torch.nn.Module) -> Scorer:
    @torch.no_grad()
    def score(preprocessed_obs: rlt.PreprocessedState) -> torch.Tensor:
        action_names, q_values = q_network(preprocessed_obs)
        return q_values

    return score


def parametric_dqn_scorer(num_actions: int, q_network: ModelBase) -> Scorer:
    @torch.no_grad()
    def score(preprocessed_obs: rlt.PreprocessedState) -> torch.Tensor:
        tiled_state = preprocessed_obs.repeat_interleave(repeats=num_actions, axis=0)

        actions = rlt.PreprocessedFeatureVector(float_features=torch.eye(num_actions))
        preprocessed_obs = rlt.PreprocessedStateAction(tiled_state.state, actions)

        q_network.eval()
        scores = q_network(preprocessed_obs).q_value.view(-1, num_actions)
        assert (
            scores.size(1) == num_actions
        ), f"scores size is {scores.size(0)}, num_actions is {num_actions}"
        q_network.train()
        return F.log_softmax(scores, dim=-1)

    return score
