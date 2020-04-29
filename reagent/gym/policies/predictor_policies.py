#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any

import reagent.types as rlt
import torch
from reagent.gym.policies import Policy


class ActorPredictorPolicy(Policy):
    def __init__(self, predictor):
        self.predictor = predictor

    @torch.no_grad()
    def act(self, obs: Any) -> rlt.ActorOutput:
        greedy_actions = self.predictor(obs).cpu()
        log_prob = torch.ones_like(greedy_actions)
        return rlt.ActorOutput(action=greedy_actions, log_prob=log_prob)
