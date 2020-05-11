#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import reagent.types as rlt
import torch
from reagent.gym.policies import Policy


class DiscreteDqnTorchPredictorPolicy(Policy):
    def __init__(self, predictor):
        self.predictor = predictor

    def act(self, obs: rlt.FeatureData) -> rlt.ActorOutput:
        # TODO: Why doesn't predictor take the whole preprocessed_state?
        state = obs.float_features
        action = self.predictor.policy(state=state).softmax
        assert action is not None
        # since act should return batched data
        return rlt.ActorOutput(action=torch.tensor([[action]]))

    @classmethod
    def get_action_extractor(cls):
        return lambda x: x.action.squeeze(0).cpu().item()


class ActorTorchPredictorPolicy(Policy):
    def __init__(self, predictor):
        self.predictor = predictor

    def act(self, obs: rlt.FeatureData) -> rlt.ActorOutput:
        # TODO: Why doesn't predictor take the whole preprocessed_state?
        state = obs.float_features
        actions = self.predictor.policy(states=state).greedy
        return rlt.ActorOutput(action=actions)

    @classmethod
    def get_action_extractor(cls):
        return lambda x: x.action.squeeze(0).cpu().numpy()
