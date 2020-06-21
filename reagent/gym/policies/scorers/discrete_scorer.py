#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Dict, List, Optional, Tuple

import reagent.types as rlt
import torch
from reagent.gym.preprocessors.trainer_preprocessor import get_possible_actions_for_gym
from reagent.gym.types import Scorer
from reagent.models.base import ModelBase


def discrete_dqn_scorer(q_network: ModelBase) -> Scorer:
    @torch.no_grad()
    def score(preprocessed_obs: rlt.FeatureData) -> torch.Tensor:
        q_network.eval()
        scores = q_network(preprocessed_obs)
        assert scores.dim() == 2, f"{scores.shape} isn't (batchsize, num_actions)."
        q_network.train()
        return scores

    return score


def discrete_qrdqn_scorer(q_network: ModelBase) -> Scorer:
    @torch.no_grad()
    def score(preprocessed_obs: rlt.FeatureData) -> torch.Tensor:
        q_network.eval()
        scores = q_network(preprocessed_obs)
        assert (
            scores.dim() == 3
        ), f"{scores.shape} isn't (batchsize, num_actions, num_atoms)."
        scores = scores.mean(dim=2)
        q_network.train()
        return scores

    return score


def discrete_dqn_serving_scorer(q_network: torch.nn.Module) -> Scorer:
    @torch.no_grad()
    def score(
        state_with_presence: Tuple[torch.Tensor, torch.Tensor],
        state_id_list_features: Optional[
            Dict[int, Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        state_id_score_list_features: Optional[
            Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> torch.Tensor:
        action_names, q_values = q_network(
            state_with_presence,
            state_id_list_features or {},
            state_id_score_list_features or {},
        )
        return q_values

    return score


def get_parametric_input(max_num_actions: int, obs: rlt.FeatureData):
    assert (
        len(obs.float_features.shape) == 2
    ), f"{obs.float_features.shape} is not (batch_size, state_dim)."
    batch_size, _ = obs.float_features.shape
    possible_actions = get_possible_actions_for_gym(batch_size, max_num_actions).to(
        obs.float_features.device
    )
    return obs.get_tiled_batch(max_num_actions), possible_actions


def parametric_dqn_scorer(max_num_actions: int, q_network: ModelBase) -> Scorer:
    @torch.no_grad()
    def score(preprocessed_obs: rlt.FeatureData) -> torch.Tensor:
        tiled_state, possible_actions = get_parametric_input(
            max_num_actions, preprocessed_obs
        )
        q_network.eval()
        scores = q_network(tiled_state, possible_actions)
        q_network.train()
        return scores.view(-1, max_num_actions)

    return score


def parametric_dqn_serving_scorer(
    max_num_actions: int, q_network: torch.nn.Module
) -> Scorer:
    @torch.no_grad()
    def score(value_presence: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        obs = value_presence[0]
        tiled_state, possible_actions = get_parametric_input(
            max_num_actions, rlt.FeatureData(obs)
        )
        tiled_state_tensor = tiled_state.float_features
        possible_actions_tensor = possible_actions.float_features
        action_names, q_values = q_network(
            state_with_presence=(
                tiled_state_tensor,
                torch.ones_like(tiled_state_tensor),
            ),
            action_with_presence=(
                possible_actions_tensor,
                torch.ones_like(possible_actions_tensor),
            ),
        )
        return q_values.view(-1, max_num_actions)

    return score
