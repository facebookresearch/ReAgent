#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from reagent.models.base import ModelBase
from reagent.preprocessing.preprocessor import Preprocessor


def split_features(
    state_and_action_with_presence: Tuple[torch.Tensor, torch.Tensor],
    state_feat_num: int,
    action_feat_num: int,
):
    # pyre-fixme[16]: `Tensor` has no attribute `narrow`.
    state_value = state_and_action_with_presence[0].narrow(1, 0, state_feat_num)
    state_presence = state_and_action_with_presence[1].narrow(1, 0, state_feat_num)
    action_value = state_and_action_with_presence[0].narrow(
        1, state_feat_num, action_feat_num
    )
    action_presence = state_and_action_with_presence[1].narrow(
        1, state_feat_num, action_feat_num
    )
    return (state_value, state_presence), (action_value, action_presence)


class SyntheticRewardPredictorWrapper(nn.Module):
    def __init__(
        self,
        seq_len: int,
        state_preprocessor: Preprocessor,
        action_preprocessor: Preprocessor,
        net: ModelBase,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.state_preprocessor = state_preprocessor
        self.action_preprocessor = action_preprocessor
        self.net = net
        self.state_feat_num = len(state_preprocessor.sorted_features)
        self.action_feat_num = len(action_preprocessor.sorted_features)

    def forward(
        self,
        state_and_action_with_presence: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        assert self.seq_len == state_and_action_with_presence[0].shape[0]
        state_with_presence, action_with_presence = split_features(
            state_and_action_with_presence,
            self.state_feat_num,
            self.action_feat_num,
        )
        # shape: seq_len, 1, state_feat_dim
        preprocessed_state = self.state_preprocessor(
            state_with_presence[0], state_with_presence[1]
        ).unsqueeze(1)
        # shape: seq_len, 1, action_feat_dim
        preprocessed_action = self.action_preprocessor(
            action_with_presence[0], action_with_presence[1]
        ).unsqueeze(1)
        # shape: (seq_len, )
        reward = self.net(preprocessed_state, preprocessed_action).flatten()
        return reward
