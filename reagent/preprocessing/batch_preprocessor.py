#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from reagent.core import types as rlt
from reagent.preprocessing.preprocessor import Preprocessor


class BatchPreprocessor(nn.Module):
    pass


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    out = {}
    for k in batch:
        out[k] = batch[k].to(device)
    return out


class DiscreteDqnBatchPreprocessor(BatchPreprocessor):
    def __init__(
        self, num_actions: int, state_preprocessor: Preprocessor, use_gpu: bool = False
    ):
        super().__init__()
        self.num_actions = num_actions
        self.state_preprocessor = state_preprocessor
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

    def forward(self, batch: Dict[str, torch.Tensor]) -> rlt.DiscreteDqnInput:
        batch = batch_to_device(batch, self.device)
        preprocessed_state = self.state_preprocessor(
            batch["state_features"], batch["state_features_presence"]
        )
        preprocessed_next_state = self.state_preprocessor(
            batch["next_state_features"], batch["next_state_features_presence"]
        )
        # not terminal iff at least one possible for next action
        not_terminal = batch["possible_next_actions_mask"].max(dim=1)[0].float()
        action = F.one_hot(batch["action"].to(torch.int64), self.num_actions)
        # next action can potentially have value self.num_action if not available
        next_action = F.one_hot(
            batch["next_action"].to(torch.int64), self.num_actions + 1
        )[:, : self.num_actions]
        return rlt.DiscreteDqnInput(
            state=rlt.FeatureData(preprocessed_state),
            next_state=rlt.FeatureData(preprocessed_next_state),
            action=action,
            next_action=next_action,
            reward=batch["reward"].unsqueeze(1),
            time_diff=batch["time_diff"].unsqueeze(1),
            step=batch["step"].unsqueeze(1),
            not_terminal=not_terminal.unsqueeze(1),
            possible_actions_mask=batch["possible_actions_mask"],
            possible_next_actions_mask=batch["possible_next_actions_mask"],
            extras=rlt.ExtraData(
                mdp_id=batch["mdp_id"].unsqueeze(1),
                sequence_number=batch["sequence_number"].unsqueeze(1),
                action_probability=batch["action_probability"].unsqueeze(1),
            ),
        )


class ParametricDqnBatchPreprocessor(BatchPreprocessor):
    def __init__(
        self,
        state_preprocessor: Preprocessor,
        action_preprocessor: Preprocessor,
        use_gpu: bool,
    ):
        super().__init__()
        self.state_preprocessor = state_preprocessor
        self.action_preprocessor = action_preprocessor
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

    def forward(self, batch: Dict[str, torch.Tensor]) -> rlt.ParametricDqnInput:
        batch = batch_to_device(batch, self.device)
        # first preprocess state and action
        preprocessed_state = self.state_preprocessor(
            batch["state_features"], batch["state_features_presence"]
        )
        preprocessed_next_state = self.state_preprocessor(
            batch["next_state_features"], batch["next_state_features_presence"]
        )
        preprocessed_action = self.action_preprocessor(
            batch["action"], batch["action_presence"]
        )
        preprocessed_next_action = self.action_preprocessor(
            batch["next_action"], batch["next_action_presence"]
        )
        return rlt.ParametricDqnInput(
            state=rlt.FeatureData(preprocessed_state),
            next_state=rlt.FeatureData(preprocessed_next_state),
            action=rlt.FeatureData(preprocessed_action),
            next_action=rlt.FeatureData(preprocessed_next_action),
            reward=batch["reward"].unsqueeze(1),
            time_diff=batch["time_diff"].unsqueeze(1),
            step=batch["step"].unsqueeze(1),
            not_terminal=batch["not_terminal"].unsqueeze(1),
            possible_actions=batch["possible_actions"],
            possible_actions_mask=batch["possible_actions_mask"],
            possible_next_actions=batch["possible_next_actions"],
            possible_next_actions_mask=batch["possible_next_actions_mask"],
            extras=rlt.ExtraData(
                mdp_id=batch["mdp_id"].unsqueeze(1),
                sequence_number=batch["sequence_number"].unsqueeze(1),
                action_probability=batch["action_probability"].unsqueeze(1),
            ),
        )


class PolicyNetworkBatchPreprocessor(BatchPreprocessor):
    def __init__(
        self,
        state_preprocessor: Preprocessor,
        action_preprocessor: Preprocessor,
        use_gpu: bool = False,
    ):
        super().__init__()
        self.state_preprocessor = state_preprocessor
        self.action_preprocessor = action_preprocessor
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

    def forward(self, batch: Dict[str, torch.Tensor]) -> rlt.PolicyNetworkInput:
        batch = batch_to_device(batch, self.device)
        preprocessed_state = self.state_preprocessor(
            batch["state_features"], batch["state_features_presence"]
        )
        preprocessed_next_state = self.state_preprocessor(
            batch["next_state_features"], batch["next_state_features_presence"]
        )
        preprocessed_action = self.action_preprocessor(
            batch["action"], batch["action_presence"]
        )
        preprocessed_next_action = self.action_preprocessor(
            batch["next_action"], batch["next_action_presence"]
        )
        return rlt.PolicyNetworkInput(
            state=rlt.FeatureData(preprocessed_state),
            next_state=rlt.FeatureData(preprocessed_next_state),
            action=rlt.FeatureData(preprocessed_action),
            next_action=rlt.FeatureData(preprocessed_next_action),
            reward=batch["reward"].unsqueeze(1),
            time_diff=batch["time_diff"].unsqueeze(1),
            step=batch["step"].unsqueeze(1),
            not_terminal=batch["not_terminal"].unsqueeze(1),
            extras=rlt.ExtraData(
                mdp_id=batch["mdp_id"].unsqueeze(1),
                sequence_number=batch["sequence_number"].unsqueeze(1),
                action_probability=batch["action_probability"].unsqueeze(1),
            ),
        )
