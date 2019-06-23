#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Tuple

import numpy as np
import torch
from ml.rl.caffe_utils import masked_softmax
from ml.rl.types import DqnPolicyActionSet, SacPolicyActionSet


logger = logging.getLogger(__name__)


class OnPolicyPredictor(object):
    """
    This class generates actions given a trainer and a state.  It's used for
    on-policy learning.  If you have a TorchScript (i.e. serialized) model,
    Use the classes in off_policy_predictor.py
    """

    def __init__(self, trainer, action_dim: int):
        self.trainer = trainer
        self.action_dim = action_dim

    def policy_net(self) -> bool:
        """
        Return True if this predictor is for a policy network
        """
        raise NotImplementedError()

    def discrete_action(self) -> bool:
        """
        Return True if this predictor is for a discrete action network
        """
        raise NotImplementedError()


class DiscreteDQNOnPolicyPredictor(OnPolicyPredictor):
    def policy(
        self, state: torch.Tensor, possible_actions_presence: torch.Tensor
    ) -> DqnPolicyActionSet:
        assert state.size()[0] == 1, "Only pass in one state when getting a policy"
        q_scores = self.predict(state)
        assert q_scores.shape[0] == 1

        # set impossible actions so low that they can't be picked
        q_scores -= (1.0 - possible_actions_presence) * 1e10  # type: ignore

        q_scores_softmax = masked_softmax(
            q_scores, possible_actions_presence, self.trainer.rl_temperature
        ).numpy()[0]
        if np.isnan(q_scores_softmax).any() or np.max(q_scores_softmax) < 1e-3:
            q_scores_softmax[:] = 1.0 / q_scores_softmax.shape[0]
        return DqnPolicyActionSet(
            greedy=int(torch.argmax(q_scores)),
            softmax=int(np.random.choice(q_scores.size()[1], p=q_scores_softmax)),
        )

    def predict(self, state):
        return self.trainer.internal_prediction(state)

    def estimate_reward(self, state):
        return self.trainer.internal_reward_estimation(state)

    def policy_net(self) -> bool:
        return False

    def discrete_action(self) -> bool:
        return True


class ParametricDQNOnPolicyPredictor(OnPolicyPredictor):
    def policy(
        self,
        states: torch.Tensor,
        possible_actions_with_presence: Tuple[torch.Tensor, torch.Tensor],
    ):
        possible_actions, possible_actions_presence = possible_actions_with_presence
        assert states.size()[0] == 1
        assert possible_actions.size()[1] == self.action_dim
        assert possible_actions.size()[0] == possible_actions_presence.size()[0]

        q_scores = self.predict(states, possible_actions)

        # set impossible actions so low that they can't be picked
        q_scores -= (
            1.0 - possible_actions_presence.reshape(1, -1)  # type: ignore
        ) * 1e10

        q_scores_softmax_numpy = masked_softmax(
            q_scores.reshape(1, -1),
            possible_actions_presence.reshape(1, -1),
            self.trainer.rl_temperature,
        ).numpy()[0]
        if (
            np.isnan(q_scores_softmax_numpy).any()
            or np.max(q_scores_softmax_numpy) < 1e-3
        ):
            q_scores_softmax_numpy[:] = 1.0 / q_scores_softmax_numpy.shape[0]
        return DqnPolicyActionSet(
            greedy=int(torch.argmax(q_scores)),
            softmax=int(np.random.choice(q_scores.size()[1], p=q_scores_softmax_numpy)),
        )

    def predict(self, state: torch.Tensor, possible_actions: torch.Tensor):
        num_actions = self.action_dim
        state_tiled = torch.repeat_interleave(state, repeats=num_actions, axis=0)
        input = (state_tiled, possible_actions)
        q_scores = self.trainer.internal_prediction(*input).reshape(1, -1)
        return q_scores

    def estimate_reward(self, states: torch.Tensor, possible_actions: torch.Tensor):
        num_actions = self.action_dim
        states_tiled = torch.repeat_interleave(states, repeats=num_actions, axis=0)
        input = (states_tiled, possible_actions)
        reward_estimates = self.trainer.internal_reward_estimation(*input).reshape(
            -1, num_actions
        )
        return reward_estimates

    def policy_net(self) -> bool:
        return False

    def discrete_action(self) -> bool:
        return False


class ContinuousActionOnPolicyPredictor(OnPolicyPredictor):
    def policy(self, states: torch.Tensor) -> SacPolicyActionSet:
        actions = self.trainer.internal_prediction(states)
        return SacPolicyActionSet(greedy=actions.cpu(), greedy_propensity=1.0)

    def policy_net(self) -> bool:
        return True
