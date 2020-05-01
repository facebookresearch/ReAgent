#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional

import numpy as np
import reagent.types as rlt
import torch
from reagent.gym.policies import Policy


class DiscreteDqnTorchPredictorPolicy(Policy):
    def __init__(self, predictor):
        self.predictor = predictor

    def act(
        self,
        preprocessed_state: rlt.PreprocessedState,
        possible_actions_mask: Optional[torch.Tensor] = None,
    ) -> int:
        state = preprocessed_state.state.float_features
        action = self.predictor.policy(
            state=state, possible_actions_presence=possible_actions_mask
        ).softmax
        assert action is not None
        # since act should return batched data
        return torch.tensor([[action]])

    @classmethod
    def get_action_extractor(cls):
        return lambda x: x.item()


class ActorTorchPredictorPolicy(Policy):
    def __init__(self, predictor):
        self.predictor = predictor

    def act(
        self,
        preprocessed_state: rlt.PreprocessedState,
        possible_actions_mask: Optional[torch.Tensor] = None,
    ) -> int:
        state = preprocessed_state.state.float_features
        actions = self.predictor.policy(states=state).greedy
        return actions

    @classmethod
    def get_action_extractor(cls):
        return lambda x: np.array([x.item()])
