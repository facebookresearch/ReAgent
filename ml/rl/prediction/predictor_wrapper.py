#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import List, Optional, Tuple

import torch
from ml.rl.preprocessing.preprocessor import Preprocessor
from torch import nn


logger = logging.getLogger(__name__)


class DiscreteDqnPredictorWrapper(torch.jit.ScriptModule):
    __constants__ = ["state_sorted_features_t"]

    def __init__(
        self,
        state_preprocessor: Preprocessor,
        value_network: Optional[nn.Module],
        action_names: List[str],
    ) -> None:
        super().__init__()

        self.state_sorted_features_t = state_preprocessor.sorted_features

        self.state_preprocessor = torch.jit.trace(
            state_preprocessor, (state_preprocessor.input_prototype())
        )

        value_network_sample_input = self.state_preprocessor(
            *state_preprocessor.input_prototype()
        )
        self.value_network = torch.jit.trace(value_network, value_network_sample_input)
        self.action_names = torch.jit.Attribute(action_names, List[str])

    @torch.jit.script_method
    def state_sorted_features(self) -> List[int]:
        return self.state_sorted_features_t

    @torch.jit.script_method
    def forward(
        self, state_with_presence: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[List[str], torch.Tensor]:
        preprocessed_state = self.state_preprocessor(
            state_with_presence[0], state_with_presence[1]
        )

        value = self.value_network(preprocessed_state)
        return (self.action_names, value)


class ParametricDqnPredictorWrapper(torch.jit.ScriptModule):
    __constants__ = ["state_sorted_features_t", "action_sorted_features_t"]

    def __init__(
        self,
        state_preprocessor: Preprocessor,
        action_preprocessor: Preprocessor,
        value_network: Optional[nn.Module],
    ) -> None:
        super().__init__()

        self.state_sorted_features_t = state_preprocessor.sorted_features
        self.state_preprocessor = torch.jit.trace(
            state_preprocessor, (state_preprocessor.input_prototype())
        )

        self.action_sorted_features_t = action_preprocessor.sorted_features
        self.action_preprocessor = torch.jit.trace(
            action_preprocessor, (action_preprocessor.input_prototype())
        )

        value_network_sample_input = (
            self.state_preprocessor(*state_preprocessor.input_prototype()),
            self.action_preprocessor(*action_preprocessor.input_prototype()),
        )
        self.value_network = torch.jit.trace(value_network, value_network_sample_input)

    @torch.jit.script_method
    def state_sorted_features(self) -> List[int]:
        return self.state_sorted_features_t

    @torch.jit.script_method
    def action_sorted_features(self) -> List[int]:
        return self.action_sorted_features_t

    @torch.jit.script_method
    def forward(
        self,
        state_with_presence: Tuple[torch.Tensor, torch.Tensor],
        action_with_presence: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[List[str], torch.Tensor]:
        preprocessed_state = self.state_preprocessor(
            state_with_presence[0], state_with_presence[1]
        )
        preprocessed_action = self.action_preprocessor(
            action_with_presence[0], action_with_presence[1]
        )

        value = self.value_network(preprocessed_state, preprocessed_action)
        return (["Q"], value)
