#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import reagent.types as rlt
import torch
from reagent.gym.policies import Policy


class ActorPredictorPolicy(Policy):
    def __init__(self, predictor):
        self.predictor = predictor

    @torch.no_grad()
    def act(self, obs: rlt.FeatureData) -> rlt.ActorOutput:
        action = self.predictor(obs).cpu()
        # TODO: return log_probs as well
        return rlt.ActorOutput(action=action)
