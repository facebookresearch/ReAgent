#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ml.rl.caffe_utils import masked_softmax
from ml.rl.preprocessing.sparse_to_dense import PythonSparseToDenseProcessor
from ml.rl.types import DqnPolicyActionSet


logger = logging.getLogger(__name__)


class DiscreteDqnTorchPredictor:
    def __init__(self, model) -> None:
        self.model = model
        self.internal_sparse_to_dense = PythonSparseToDenseProcessor(
            self.model.state_sorted_features()
        )
        self.softmax_temperature: Optional[float] = None

    def predict(self, state_features: List[Dict[int, float]]) -> List[Dict[str, float]]:
        dense_state_features, dense_state_feature_exist_mask = self.internal_sparse_to_dense(
            state_features
        )
        action_names, values = self.model(
            (dense_state_features, dense_state_feature_exist_mask)
        )
        retval = []
        for i in range(values.size()[0]):
            retval_item: Dict[str, float] = {}
            for j, action in enumerate(action_names):
                retval_item[action] = values[i][j]
            retval.append(retval_item)
        return retval

    def policy(
        self,
        state: torch.Tensor,
        state_feature_presence: Optional[torch.Tensor] = None,
        possible_actions_presence: Optional[torch.Tensor] = None,
    ) -> DqnPolicyActionSet:
        assert state.size()[0] == 1, "Only pass in one state when getting a policy"
        assert (
            self.softmax_temperature is not None
        ), "Please set the softmax temperature before calling policy()"

        if state_feature_presence is None:
            state_feature_presence = torch.ones_like(state)
        action_names, q_scores = self.model((state, state_feature_presence))

        return self.policy_given_q_values(
            q_scores, action_names, self.softmax_temperature, possible_actions_presence
        )

    @staticmethod
    def policy_given_q_values(
        q_scores: torch.Tensor,
        action_names: List[str],
        softmax_temperature: float,
        possible_actions_presence: Optional[torch.Tensor] = None,
    ) -> DqnPolicyActionSet:
        assert q_scores.shape[0] == 1 and len(q_scores.shape) == 2

        if possible_actions_presence is None:
            possible_actions_presence = torch.ones_like(q_scores)
        possible_actions_presence = possible_actions_presence.reshape(1, -1)
        assert possible_actions_presence.shape == q_scores.shape

        # set impossible actions so low that they can't be picked
        q_scores -= (1.0 - possible_actions_presence) * 1e10  # type: ignore

        q_scores_softmax = (
            masked_softmax(q_scores, possible_actions_presence, softmax_temperature)
            .detach()
            .numpy()[0]
        )
        if np.isnan(q_scores_softmax).any() or np.max(q_scores_softmax) < 1e-3:
            q_scores_softmax[:] = 1.0 / q_scores_softmax.shape[0]
        greedy_act_idx = int(torch.argmax(q_scores))
        softmax_act_idx = int(np.random.choice(q_scores.size()[1], p=q_scores_softmax))

        return DqnPolicyActionSet(
            greedy=greedy_act_idx,
            softmax=softmax_act_idx,
            greedy_act_name=action_names[greedy_act_idx],
            softmax_act_name=action_names[softmax_act_idx],
        )

    def policy_net(self) -> bool:
        return False

    def discrete_action(self) -> bool:
        return True


class ParametricDqnTorchPredictor:
    def __init__(self, model) -> None:
        self.model = model
        self.state_internal_sparse_to_dense = PythonSparseToDenseProcessor(
            self.model.state_sorted_features()
        )
        self.action_internal_sparse_to_dense = PythonSparseToDenseProcessor(
            self.model.action_sorted_features()
        )
        self.softmax_temperature: Optional[float] = None

    def predict(
        self,
        state_features: List[Dict[int, float]],
        action_features: List[Dict[int, float]],
    ) -> List[Dict[str, float]]:
        dense_state_features, dense_state_feature_exist_mask = self.state_internal_sparse_to_dense(
            state_features
        )
        dense_action_features, dense_action_feature_exist_mask = self.action_internal_sparse_to_dense(
            action_features
        )
        action_names, values = self.model(
            (dense_state_features, dense_state_feature_exist_mask),
            (dense_action_features, dense_action_feature_exist_mask),
        )
        retval = []
        for i in range(values.size()[0]):
            retval_item: Dict[str, float] = {}
            for j, action in enumerate(action_names):
                retval_item[action] = values[i][j]
            retval.append(retval_item)
        return retval

    def policy(
        self,
        tiled_states: torch.Tensor,
        possible_actions_with_presence: Tuple[torch.Tensor, torch.Tensor],
    ):
        possible_actions, possible_actions_presence = possible_actions_with_presence
        assert tiled_states.size()[0] == possible_actions.size()[0]
        assert possible_actions.size()[0] == possible_actions_presence.size()[0]
        assert (
            self.softmax_temperature is not None
        ), "Please set the softmax temperature before calling policy()"

        state_feature_presence = torch.ones_like(tiled_states)
        _, q_scores = self.model(
            (tiled_states, state_feature_presence), possible_actions_with_presence
        )
        q_scores = q_scores.reshape(1, -1)

        return self.policy_given_q_values(
            q_scores, self.softmax_temperature, possible_actions_presence
        )

    @staticmethod
    def policy_given_q_values(
        q_scores: torch.Tensor,
        softmax_temperature: float,
        possible_actions_presence: torch.Tensor,
    ) -> DqnPolicyActionSet:
        assert q_scores.shape[0] == 1 and len(q_scores.shape) == 2
        possible_actions_presence = possible_actions_presence.reshape(1, -1)
        assert possible_actions_presence.shape == q_scores.shape

        # set impossible actions so low that they can't be picked
        q_scores -= (1.0 - possible_actions_presence) * 1e10

        q_scores_softmax_numpy = (
            masked_softmax(
                q_scores.reshape(1, -1), possible_actions_presence, softmax_temperature
            )
            .detach()
            .numpy()[0]
        )
        if (
            np.isnan(q_scores_softmax_numpy).any()
            or np.max(q_scores_softmax_numpy) < 1e-3
        ):
            q_scores_softmax_numpy[:] = 1.0 / q_scores_softmax_numpy.shape[0]

        return DqnPolicyActionSet(
            greedy=int(torch.argmax(q_scores)),
            softmax=int(np.random.choice(q_scores.size()[1], p=q_scores_softmax_numpy)),
        )

    def policy_net(self) -> bool:
        return False

    def discrete_action(self) -> bool:
        return False
