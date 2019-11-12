#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional

import ml.rl.types as rlt
import numpy as np
import torch
from ml.rl.models.mdn_rnn import transpose


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

    def as_cem_training_batch(self, batch_first=False):
        """
        Generate one-step samples needed by CEM trainer.
        The samples will be used to train an ensemble of world models used by CEM.

        If batch_first = True:
            state/next state shape: batch_size x 1 x state_dim
            action shape: batch_size x 1 x action_dim
            reward/terminal shape: batch_size x 1
        else (default):
             state/next state shape: 1 x batch_size x state_dim
             action shape: 1 x batch_size x action_dim
             reward/terminal shape: 1 x batch_size
        """
        if batch_first:
            seq_len_dim = 1
            reward, not_terminal = self.rewards, self.not_terminal
        else:
            seq_len_dim = 0
            reward, not_terminal = transpose(self.rewards, self.not_terminal)
        training_input = rlt.PreprocessedMemoryNetworkInput(
            state=rlt.PreprocessedFeatureVector(
                float_features=self.states.unsqueeze(seq_len_dim)
            ),
            action=self.actions.unsqueeze(seq_len_dim),
            next_state=rlt.PreprocessedFeatureVector(
                float_features=self.next_states.unsqueeze(seq_len_dim)
            ),
            reward=reward,
            not_terminal=not_terminal,
            step=self.step,
            time_diff=self.time_diffs,
        )
        return rlt.PreprocessedTrainingBatch(
            training_input=training_input,
            extras=rlt.ExtraData(
                mdp_id=self.mdp_ids,
                sequence_number=self.sequence_numbers,
                action_probability=self.propensities,
                max_num_actions=self.max_num_actions,
                metrics=self.metrics,
            ),
        )

    def as_parametric_maxq_training_batch(self):
        state_dim = self.states.shape[1]
        return rlt.PreprocessedTrainingBatch(
            training_input=rlt.PreprocessedParametricDqnInput(
                state=rlt.PreprocessedFeatureVector(float_features=self.states),
                action=rlt.PreprocessedFeatureVector(float_features=self.actions),
                next_state=rlt.PreprocessedFeatureVector(
                    float_features=self.next_states
                ),
                next_action=rlt.PreprocessedFeatureVector(
                    float_features=self.next_actions
                ),
                tiled_next_state=rlt.PreprocessedFeatureVector(
                    float_features=self.possible_next_actions_state_concat[
                        :, :state_dim
                    ]
                ),
                possible_actions=None,
                possible_actions_mask=self.possible_actions_mask,
                possible_next_actions=rlt.PreprocessedFeatureVector(
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

    def as_policy_network_training_batch(self):
        return rlt.PreprocessedTrainingBatch(
            training_input=rlt.PreprocessedPolicyNetworkInput(
                state=rlt.PreprocessedFeatureVector(float_features=self.states),
                action=rlt.PreprocessedFeatureVector(float_features=self.actions),
                next_state=rlt.PreprocessedFeatureVector(
                    float_features=self.next_states
                ),
                next_action=rlt.PreprocessedFeatureVector(
                    float_features=self.next_actions
                ),
                reward=self.rewards,
                not_terminal=self.not_terminal,
                step=self.step,
                time_diff=self.time_diffs,
            ),
            extras=rlt.ExtraData(),
        )

    def as_discrete_maxq_training_batch(self):
        return rlt.PreprocessedTrainingBatch(
            training_input=rlt.PreprocessedDiscreteDqnInput(
                state=rlt.PreprocessedFeatureVector(float_features=self.states),
                action=self.actions,
                next_state=rlt.PreprocessedFeatureVector(
                    float_features=self.next_states
                ),
                next_action=self.next_actions,
                possible_actions_mask=self.possible_actions_mask,
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

    def as_slate_q_training_batch(self):
        batch_size, state_dim = self.states.shape
        action_dim = self.actions.shape[1]
        num_actions = self.possible_actions_mask.shape[1]
        return rlt.PreprocessedTrainingBatch(
            training_input=rlt.PreprocessedSlateQInput(
                state=rlt.PreprocessedFeatureVector(float_features=self.states),
                next_state=rlt.PreprocessedFeatureVector(
                    float_features=self.next_states
                ),
                tiled_state=rlt.PreprocessedTiledFeatureVector(
                    float_features=self.possible_actions_state_concat[
                        :, :state_dim
                    ].view(batch_size, -1, state_dim)
                ),
                tiled_next_state=rlt.PreprocessedTiledFeatureVector(
                    float_features=self.possible_next_actions_state_concat[
                        :, :state_dim
                    ].view(batch_size, -1, state_dim)
                ),
                ###
                action=rlt.PreprocessedSlateFeatureVector(
                    float_features=self.possible_actions_state_concat[
                        :, state_dim:
                    ].view(batch_size, -1, action_dim),
                    # HACK: this should be used only in SlateQ test
                    item_mask=self.possible_next_actions_mask,
                    # FIXME: need true probability
                    item_probability=torch.full_like(
                        self.possible_next_actions_mask,
                        1.0 / num_actions,
                        dtype=torch.float,
                    ),
                ),
                next_action=rlt.PreprocessedSlateFeatureVector(
                    float_features=self.possible_next_actions_state_concat[
                        :, state_dim:
                    ].view(batch_size, -1, action_dim),
                    item_mask=self.possible_next_actions_mask,
                    # FIXME: need true probability
                    item_probability=torch.full_like(
                        self.possible_next_actions_mask,
                        1.0 / num_actions,
                        dtype=torch.float,
                    ),
                ),
                # HACK: This is ok since reward will be masked
                reward=self.rewards.repeat_interleave(num_actions, dim=1),
                # HACK: this should be used only in SlateQ test
                reward_mask=self.possible_actions_mask,
                ####
                time_diff=self.time_diffs,
                step=self.step,
                not_terminal=self.not_terminal,
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

    def set_device(self, device):
        for x in TrainingDataPage.__slots__:
            if x in ("mdp_ids", "sequence_numbers", "max_num_actions"):
                continue  # Torch does not support tensors of strings
            t = getattr(self, x)
            if t is not None:
                assert isinstance(t, torch.Tensor), (
                    x + " is not a torch tensor (is " + str(type(t)) + ")"
                )
                if x == "possible_next_actions_lengths":
                    setattr(self, x, t.to(device=device).long())
                else:
                    setattr(self, x, t.to(device=device).float())
