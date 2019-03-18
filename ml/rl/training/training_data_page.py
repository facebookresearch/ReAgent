#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional

import ml.rl.types as rlt
import numpy as np
import torch


class TrainingDataPage(object):
    __slots__ = [
        "mdp_ids",
        "sequence_numbers",
        "states",
        "actions",
        "propensities",
        "rewards",
        "possible_actions_state_concat",
        "possible_actions_mask",
        "next_states",
        "next_actions",
        "possible_next_actions_state_concat",
        "possible_next_actions_mask",
        "not_terminal",
        "time_diffs",
        "metrics",
        "step",
        "max_num_actions",
    ]

    def __init__(
        self,
        mdp_ids: Optional[np.ndarray] = None,
        sequence_numbers: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        propensities: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        possible_actions_mask: Optional[torch.Tensor] = None,
        possible_actions_state_concat: Optional[torch.Tensor] = None,
        next_states: Optional[torch.Tensor] = None,
        next_actions: Optional[torch.Tensor] = None,
        possible_next_actions_mask: Optional[torch.Tensor] = None,
        possible_next_actions_state_concat: Optional[torch.Tensor] = None,
        not_terminal: Optional[torch.Tensor] = None,
        time_diffs: Optional[torch.Tensor] = None,
        metrics: Optional[torch.Tensor] = None,
        step: Optional[torch.Tensor] = None,
        max_num_actions: Optional[int] = None,
    ) -> None:
        """
        Creates a TrainingDataPage object.

        In the case where `not_terminal` can be determined by next_actions or
        possible_next_actions, feel free to omit it.
        """
        self.mdp_ids = mdp_ids
        self.sequence_numbers = sequence_numbers
        self.states = states
        self.actions = actions
        self.propensities = propensities
        self.rewards = rewards
        self.possible_actions_mask = possible_actions_mask
        self.possible_actions_state_concat = possible_actions_state_concat
        self.next_states = next_states
        self.next_actions = next_actions
        self.not_terminal = not_terminal
        self.time_diffs = time_diffs
        self.possible_next_actions_mask = possible_next_actions_mask
        self.possible_next_actions_state_concat = possible_next_actions_state_concat
        self.metrics = metrics
        self.step = step
        self.max_num_actions = max_num_actions

    def as_parametric_sarsa_training_batch(self):
        return rlt.TrainingBatch(
            training_input=rlt.SARSAInput(
                state=rlt.FeatureVector(float_features=self.states),
                action=rlt.FeatureVector(float_features=self.actions),
                next_state=rlt.FeatureVector(float_features=self.next_states),
                next_action=rlt.FeatureVector(float_features=self.next_actions),
                reward=self.rewards,
                not_terminal=self.not_terminal,
                step=self.step,
                time_diff=self.time_diffs,
            ),
            extras=rlt.ExtraData(),
        )

    def as_parametric_maxq_training_batch(self):
        state_dim = self.states.shape[1]
        return rlt.TrainingBatch(
            training_input=rlt.MaxQLearningInput(
                state=rlt.FeatureVector(float_features=self.states),
                action=rlt.FeatureVector(float_features=self.actions),
                next_state=None,
                next_action=None,
                tiled_next_state=rlt.FeatureVector(
                    float_features=self.possible_next_actions_state_concat[
                        :, :state_dim
                    ]
                ),
                possible_actions=None,
                possible_actions_mask=self.possible_actions_mask,
                possible_next_actions=rlt.FeatureVector(
                    float_features=self.possible_next_actions_state_concat[
                        :, state_dim:
                    ]
                ),
                possible_next_actions_mask=self.possible_next_actions_mask,
                reward=self.rewards,
                not_terminal=self.not_terminal,
                step=self.step,
                time_diff=self.time_diffs,
            ),
            extras=rlt.ExtraData(),
        )

    def as_discrete_sarsa_training_batch(self):
        return rlt.TrainingBatch(
            training_input=rlt.SARSAInput(
                state=rlt.FeatureVector(float_features=self.states),
                action=self.actions,
                next_state=rlt.FeatureVector(float_features=self.next_states),
                next_action=self.next_actions,
                reward=self.rewards,
                not_terminal=self.not_terminal,
                step=self.step,
                time_diff=self.time_diffs,
            ),
            extras=rlt.ExtraData(
                mdp_id=self.mdp_ids,
                sequence_number=self.sequence_numbers,
                action_probability=self.propensities,
                max_num_actions=self.max_num_actions,
                metrics=self.metrics,
            ),
        )

    def as_discrete_maxq_training_batch(self):
        return rlt.TrainingBatch(
            training_input=rlt.MaxQLearningInput(
                state=rlt.FeatureVector(float_features=self.states),
                action=self.actions,
                next_state=rlt.FeatureVector(float_features=self.next_states),
                next_action=self.next_actions,
                tiled_next_state=None,
                possible_actions=None,
                possible_actions_mask=self.possible_actions_mask,
                possible_next_actions=None,
                possible_next_actions_mask=self.possible_next_actions_mask,
                reward=self.rewards,
                not_terminal=self.not_terminal,
                step=self.step,
                time_diff=self.time_diffs,
            ),
            extras=rlt.ExtraData(
                mdp_id=self.mdp_ids,
                sequence_number=self.sequence_numbers,
                action_probability=self.propensities,
                max_num_actions=self.max_num_actions,
                metrics=self.metrics,
            ),
        )

    def size(self) -> int:
        if self.states:
            return len(self.states)
        raise Exception("Cannot get size of TrainingDataPage missing states.")

    def set_type(self, dtype):
        # TODO: Clean this up in a future diff.  Figure out which should be long/float
        for x in TrainingDataPage.__slots__:
            if x in ("mdp_ids", "sequence_numbers", "max_num_actions"):
                continue  # Torch does not support tensors of strings
            t = getattr(self, x)
            if t is not None:
                assert isinstance(t, torch.Tensor), (
                    x + " is not a torch tensor (is " + str(type(t)) + ")"
                )
                if x == "possible_next_actions_lengths":
                    setattr(self, x, t.type(dtype).long())
                else:
                    setattr(self, x, t.type(dtype))
