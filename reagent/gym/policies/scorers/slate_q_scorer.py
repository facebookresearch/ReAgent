#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.gym.types import Scorer
from reagent.models.base import ModelBase


def slate_q_scorer(num_candidates: int, q_network: ModelBase) -> Scorer:
    @torch.no_grad()
    def score(state: rlt.FeatureData) -> torch.Tensor:
        tiled_state = state.repeat_interleave(repeats=num_candidates, axis=0)
        candidate_docs = state.candidate_docs
        assert candidate_docs is not None
        actions = candidate_docs.as_feature_data()

        q_network.eval()
        scores = q_network(tiled_state, actions).view(-1, num_candidates)
        q_network.train()

        select_prob = F.softmax(candidate_docs.value, dim=1)
        assert select_prob.shape == scores.shape

        return select_prob * scores

    return score


def slate_q_serving_scorer(num_candidates: int, q_network: torch.nn.Module) -> Scorer:
    @torch.no_grad()
    def score(state: rlt.FeatureData) -> torch.Tensor:
        # pyre-fixme[16]: `Tensor` has no attribute `repeat_interleave`.
        tiled_state = state.float_features.repeat_interleave(
            repeats=num_candidates, axis=0
        )
        candidate_docs = state.candidate_docs
        assert candidate_docs is not None
        actions = candidate_docs.as_feature_data().float_features

        q_network.eval()
        action_names, q_values = q_network(
            (tiled_state, torch.ones_like(tiled_state)),
            (actions, torch.ones_like(actions)),
        )
        scores = q_values.view(-1, num_candidates)
        q_network.train()

        select_prob = F.softmax(candidate_docs.value, dim=1)
        assert select_prob.shape == scores.shape

        return select_prob * scores

    return score
