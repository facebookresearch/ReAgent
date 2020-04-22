#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
Preprocess a batch of transitions sampled from the ReplayBuffer for
Trainer.train, which we expect accepts rlt.PreprocessedTrainingBatch.
"""

from typing import Any, Dict

import torch
from reagent import types as rlt
from reagent.gym.types import TrainerPreprocessor
from reagent.parameters import NormalizationParameters


def parametric_dqn_trainer_preprocessor(
    num_actions: int, state_normalization: Dict[int, NormalizationParameters]
) -> TrainerPreprocessor:
    def preprocess_batch(train_batch: Any) -> rlt.PreprocessedTrainingBatch:
        obs, action, reward, next_obs, next_action, next_reward, terminal, idxs, possible_actions_mask, log_prob = (
            train_batch
        )
        batch_size = obs.shape[0]

        obs = torch.tensor(obs).squeeze(2)
        action = torch.tensor(action).float()
        next_obs = torch.tensor(next_obs).squeeze(2)
        next_action = torch.tensor(next_action).to(torch.float32)
        reward = torch.tensor(reward).unsqueeze(1)
        not_terminal = 1 - torch.tensor(terminal).unsqueeze(1).to(torch.uint8)
        possible_actions_mask = torch.ones_like(action).to(torch.bool)

        tiled_next_state = torch.repeat_interleave(
            next_obs, repeats=num_actions, axis=0
        )
        possible_next_actions = torch.eye(num_actions).repeat(batch_size, 1)
        possible_next_actions_mask = not_terminal.repeat(1, num_actions).to(torch.bool)
        return rlt.PreprocessedTrainingBatch(
            rlt.PreprocessedParametricDqnInput(
                state=rlt.PreprocessedFeatureVector(float_features=obs),
                action=rlt.PreprocessedFeatureVector(float_features=action),
                next_state=rlt.PreprocessedFeatureVector(float_features=next_obs),
                next_action=rlt.PreprocessedFeatureVector(float_features=next_action),
                possible_actions=None,
                possible_actions_mask=possible_actions_mask,
                possible_next_actions=rlt.PreprocessedFeatureVector(
                    float_features=possible_next_actions
                ),
                possible_next_actions_mask=possible_next_actions_mask,
                tiled_next_state=rlt.PreprocessedFeatureVector(
                    float_features=tiled_next_state
                ),
                reward=reward,
                not_terminal=not_terminal,
                step=None,
                time_diff=None,
            ),
            extras=rlt.ExtraData(),
        )

    return preprocess_batch


def sac_trainer_preprocessor() -> TrainerPreprocessor:
    def preprocess_batch(train_batch: Any) -> rlt.PreprocessedTrainingBatch:
        obs, action, reward, next_obs, next_action, next_reward, terminal, idxs, possible_actions_mask, log_prob = (
            train_batch
        )
        obs = torch.tensor(obs).squeeze(2)
        action = torch.tensor(action).float()
        reward = torch.tensor(reward).unsqueeze(1)
        next_obs = torch.tensor(next_obs).squeeze(2)
        next_action = torch.tensor(next_action)
        not_terinal = 1.0 - torch.tensor(terminal).unsqueeze(1).float()
        idxs = torch.tensor(idxs)
        possible_actions_mask = torch.tensor(possible_actions_mask).float()
        log_prob = torch.tensor(log_prob)
        return rlt.PreprocessedTrainingBatch(
            training_input=rlt.PreprocessedPolicyNetworkInput(
                state=rlt.PreprocessedFeatureVector(float_features=obs),
                action=rlt.PreprocessedFeatureVector(float_features=action),
                next_state=rlt.PreprocessedFeatureVector(float_features=next_obs),
                next_action=rlt.PreprocessedFeatureVector(float_features=next_action),
                reward=reward,
                not_terminal=not_terinal,
                step=None,
                time_diff=None,
            ),
            extras=rlt.ExtraData(),
        )

    return preprocess_batch
