#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from reagent.preprocessing.sparse_to_dense import PythonSparseToDenseProcessor
from reagent.torch_utils import masked_softmax
from reagent.types import DqnPolicyActionSet, SacPolicyActionSet


logger = logging.getLogger(__name__)


class DiscreteDqnTorchPredictor:
    def __init__(self, model) -> None:
        self.model = model
        self.internal_sparse_to_dense = PythonSparseToDenseProcessor(
            self.model.state_sorted_features()
        )
        self.softmax_temperature: Optional[float] = None

    def predict(self, state_features: List[Dict[int, float]]) -> List[Dict[str, float]]:
        (
            dense_state_features,
            dense_state_feature_exist_mask,
        ) = self.internal_sparse_to_dense(state_features)
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
            q_scores,
            action_names,
            # pyre-fixme[6]: Expected `float` for 3rd param but got `Optional[float]`.
            self.softmax_temperature,
            possible_actions_presence,
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
        q_scores -= (1.0 - possible_actions_presence) * 1e10

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
            softmax_act_prob=q_scores_softmax[softmax_act_idx],
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
        (
            dense_state_features,
            dense_state_feature_exist_mask,
        ) = self.state_internal_sparse_to_dense(state_features)
        (
            dense_action_features,
            dense_action_feature_exist_mask,
        ) = self.action_internal_sparse_to_dense(action_features)
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
            q_scores,
            # pyre-fixme[6]: Expected `float` for 2nd param but got `Optional[float]`.
            self.softmax_temperature,
            torch.ones_like(q_scores),
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

        greedy_act_idx = int(torch.argmax(q_scores))
        softmax_act_idx = int(
            np.random.choice(q_scores.size()[1], p=q_scores_softmax_numpy)
        )
        return DqnPolicyActionSet(
            greedy=greedy_act_idx,
            softmax=softmax_act_idx,
            softmax_act_prob=float(q_scores_softmax_numpy[softmax_act_idx]),
        )

    def policy_net(self) -> bool:
        return False

    def discrete_action(self) -> bool:
        return False


class ActorTorchPredictor:
    def __init__(self, model, action_feature_ids: List[int]) -> None:
        self.model = model
        self.internal_sparse_to_dense = PythonSparseToDenseProcessor(
            self.model.state_sorted_features()
        )
        self.action_feature_ids = action_feature_ids

    def predict(self, state_features: List[Dict[int, float]]) -> List[Dict[str, float]]:
        (
            dense_state_features,
            dense_state_feature_exist_mask,
        ) = self.internal_sparse_to_dense(state_features)
        actions = self.model((dense_state_features, dense_state_feature_exist_mask))
        assert actions.shape[1:] == (len(self.action_feature_ids),)
        retval = [
            {str(fid): val.item() for fid, val in zip(self.action_feature_ids, action)}
            for action in actions
        ]
        return retval

    def actor_prediction(
        self, float_state_features: List[Dict[int, float]]
    ) -> List[Dict[str, float]]:
        return self.predict(float_state_features)

    def policy_net(self) -> bool:
        return True

    def policy(self, states: torch.Tensor) -> SacPolicyActionSet:
        state_masks = torch.ones_like(states, dtype=torch.bool)
        actions = self.model((states, state_masks)).detach()
        assert actions.shape[1:] == (len(self.action_feature_ids),)
        return SacPolicyActionSet(greedy=actions.cpu(), greedy_propensity=1.0)
