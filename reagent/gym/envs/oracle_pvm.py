#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from collections import OrderedDict
from typing import Callable, Dict, List

# pyre-fixme[21]: Could not find module `gym`.
import gym
import numpy as np
import reagent.types as rlt
import torch
from reagent.core.dataclasses import dataclass
from reagent.gym.envs import RecSim
from reagent.gym.preprocessors.default_preprocessors import RecsimObsPreprocessor
from scipy import stats


logger = logging.getLogger(__name__)

# score function takes user and doc features, and outputs a score
SCORE_FUNCTION_T = Callable[[np.ndarray, np.ndarray], float]


def make_default_score_fn(fn_i: int) -> SCORE_FUNCTION_T:
    """
    Make ith score_fn (constructor of ith score)
    """

    def fn(user: np.ndarray, doc: np.ndarray) -> float:
        return doc[fn_i]
        # user = user ** (fn_i + 1)
        # doc = doc ** (fn_i + 1)
        # return np.inner(user, doc)
        # return user[fn_i] * doc[fn_i]

    return fn


VM_WEIGHT_LOW = -1.0
VM_WEIGHT_HIGH = 1.0
MATCH_REWARD_BOOST = 3.0


def get_default_score_fns(num_weights):
    return [make_default_score_fn(i) for i in range(num_weights)]


def get_ground_truth_weights(num_weights):
    return np.array([1] * num_weights)


@dataclass
class OraclePVM(RecSim):
    """
    Wrapper over RecSim for simulating (Personalized) VM Tuning.
    The state is the same as for RecSim (user feature + candidate features).
    There are num_weights VM weights to tune, and so action space is a vector
    of length num_weights.
    OraclePVM hides num_weights number of
    (1) score_fns (akin to VM models), that take in
        user + candidate_i feature and produces a score for candidate_i.
    (2) ground_truth_weights, that are used to produce "ground truth", a.k.a.
        "Oracle", rankings.
    Reward is the Kendall-Tau between ground truth and the ranking created from the
    weights given by action. If the rankings match exactly, the reward is boosted to 3.
    NOTE: This environment only tests if the Agent can learn the hidden ground
    truth weights, which may be far from optimal (in terms of RecSim's rewards,
    which we're ignoring). This is easier for unit tests, but in the real world
    we will be trying to learn the optimal weights, and the reward signal would
    reflect that.

    TODO: made environment easier to learn from by not using RecSim.
    """

    user_feat_dim: int = 1
    candidate_feat_dim: int = 3
    num_weights: int = 3

    def __post_init_post_parse__(self):
        assert (
            self.slate_size == self.num_candidates
        ), f"Must be equal (slate_size) {self.slate_size} != (num_candidates) {self.num_candidates}"
        super().__post_init_post_parse__()
        self.score_fns: List[SCORE_FUNCTION_T] = get_default_score_fns(self.num_weights)
        self.ground_truth_weights: List[float] = get_ground_truth_weights(
            self.num_weights
        )
        assert len(self.score_fns) == len(
            self.ground_truth_weights
        ), f"{len(self.score_fns)} != {len(self.ground_truth_weights)}"
        assert (
            len(self.ground_truth_weights) == self.num_weights
        ), f"{self.ground_truth_weights.shape} != {self.num_weights}"

    def reset(self):
        self.prev_obs = super().reset()
        self.prev_obs.update(
            {
                "user": np.random.rand(self.user_feat_dim),
                "doc": OrderedDict(
                    [
                        (str(i), np.random.rand(self.candidate_feat_dim))
                        for i in range(self.num_candidates)
                    ]
                ),
            }
        )
        return self.prev_obs

    def step(self, action):
        user_feat = self.prev_obs["user"]
        doc_feats = self.prev_obs["doc"]
        scores = self._get_scores(user_feat, doc_feats)
        ground_truth_ranking = self._get_ranking(scores, self.ground_truth_weights)
        policy_ranking = self._get_ranking(scores, action)
        t = True
        # comment out to avoid non-stationary
        # self.prev_obs, _, t, i = super().step(policy_ranking)

        num_matches = (ground_truth_ranking == policy_ranking).sum()
        if num_matches == self.slate_size:
            reward = MATCH_REWARD_BOOST
        else:
            reward, _p_value = stats.kendalltau(ground_truth_ranking, policy_ranking)
        return self.prev_obs, reward, t, None

    def is_match(self, reward):
        # for evaluation, return true iff the reward represents a match
        return reward > (MATCH_REWARD_BOOST - 1e-6)

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=VM_WEIGHT_LOW, high=VM_WEIGHT_HIGH, shape=(self.num_weights,)
        )

    @action_space.setter
    def action_space(self, val):
        pass

    def _get_scores(
        self, user_feat: np.ndarray, doc_feats: Dict[str, np.ndarray]
    ) -> np.ndarray:
        # num_docs x num_scores where i,j coordinate is jth score for ith doc
        scores = np.array(
            [
                # pyre-fixme[16]: `OraclePVM` has no attribute `score_fns`.
                [score_fn(user_feat, doc_feat) for score_fn in self.score_fns]
                for _k, doc_feat in doc_feats.items()
            ]
        )
        return scores

    def _get_ranking(self, scores: np.ndarray, weights: np.ndarray):
        assert weights.shape == (scores.shape[1],), f"{weights.shape}, {scores.shape}"
        weighted_scores = scores * weights
        values = weighted_scores.sum(axis=1)
        indices = np.argsort(-values)
        return indices[: self.slate_size]

    def obs_preprocessor(self, obs: np.ndarray) -> rlt.FeatureData:
        preprocessor = RecsimObsPreprocessor.create_from_env(self)
        preprocessed_obs = preprocessor(obs)
        return rlt._embed_states(preprocessed_obs)

    def serving_obs_preprocessor(self, obs: np.ndarray):
        preprocessor = RecsimObsPreprocessor.create_from_env(self)
        x = preprocessor(obs)
        # user was batch_size x state_size, stack
        user = x.float_features.unsqueeze(1).repeat_interleave(
            self.num_candidates, dim=1
        )
        candidates = x.candidate_docs.float_features
        combined = torch.cat([user, candidates], dim=2).squeeze(0)
        return (combined, torch.ones_like(combined, dtype=torch.uint8))
