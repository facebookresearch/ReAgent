#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Dict, List, cast

import numpy as np
import torch
from ml.rl import types as rlt
from ml.rl.preprocessing.sparse_to_dense import SparseToDenseProcessor


class PreprocessHandler:
    def __init__(self, sparse_to_dense_processor: SparseToDenseProcessor):
        self.sparse_to_dense_processor = sparse_to_dense_processor

    def preprocess(self, batch) -> rlt.RawTrainingBatch:
        state_features_dense, state_features_dense_presence = self.sparse_to_dense_processor(
            batch["state_features"]
        )
        next_state_features_dense, next_state_features_dense_presence = self.sparse_to_dense_processor(
            batch["next_state_features"]
        )

        mdp_ids = np.array(batch["mdp_id"]).reshape(-1, 1)
        sequence_numbers = torch.tensor(
            batch["sequence_number"], dtype=torch.int32
        ).reshape(-1, 1)
        rewards = torch.tensor(batch["reward"], dtype=torch.float32).reshape(-1, 1)
        time_diffs = torch.tensor(batch["time_diff"], dtype=torch.int32).reshape(-1, 1)
        if "action_probability" in batch:
            propensities = torch.tensor(
                batch["action_probability"], dtype=torch.float32
            ).reshape(-1, 1)
        else:
            propensities = torch.ones(rewards.shape, dtype=torch.float32)

        return rlt.RawTrainingBatch(
            training_input=rlt.RawBaseInput(  # type: ignore
                state=rlt.FeatureVector(
                    float_features=rlt.ValuePresence(
                        value=state_features_dense,
                        presence=state_features_dense_presence,
                    )
                ),
                next_state=rlt.FeatureVector(
                    float_features=rlt.ValuePresence(
                        value=next_state_features_dense,
                        presence=next_state_features_dense_presence,
                    )
                ),
                reward=rewards,
                time_diff=time_diffs,
                step=None,
                not_terminal=None,
            ),
            extras=rlt.ExtraData(
                mdp_id=mdp_ids,
                sequence_number=sequence_numbers,
                action_probability=propensities,
            ),
        )


class DiscreteDqnPreprocessHandler(PreprocessHandler):
    def __init__(
        self, num_actions: int, sparse_to_dense_processor: SparseToDenseProcessor
    ):
        super().__init__(sparse_to_dense_processor)
        self.num_actions = num_actions

    def read_actions(self, actions):
        actions = np.array(actions)
        actions = np.expand_dims(actions, axis=1)
        action_names_tiled = np.tile(list(range(self.num_actions)), actions.shape)
        retval = torch.tensor(
            (actions == action_names_tiled).astype(np.float32), dtype=torch.float32
        )
        return retval

    def preprocess(self, batch) -> rlt.RawTrainingBatch:
        training_batch = super().preprocess(batch)
        actions = self.read_actions(batch["action"])
        pas_mask = torch.from_numpy(
            np.array(batch["possible_actions"], dtype=np.float32)
        ).byte()

        next_actions = self.read_actions(batch["next_action"])
        pnas_mask = np.array(batch["possible_next_actions"], dtype=np.float32)
        not_terminal = torch.max(next_actions, 1)[0].float().reshape(-1, 1)
        pnas_mask_torch = torch.from_numpy(pnas_mask).byte()

        base_input = training_batch.training_input
        training_input = rlt.RawDiscreteDqnInput(
            state=base_input.state,
            reward=base_input.reward,
            time_diff=base_input.time_diff,
            action=actions,
            next_action=next_actions,
            not_terminal=not_terminal,
            next_state=base_input.next_state,
            possible_actions_mask=pas_mask,
            possible_next_actions_mask=pnas_mask_torch,
            step=None,
        )
        return training_batch._replace(training_input=training_input)


class ParametricDqnPreprocessHandler(PreprocessHandler):
    def __init__(
        self,
        state_sparse_to_dense: SparseToDenseProcessor,
        action_sparse_to_dense: SparseToDenseProcessor,
    ):
        super().__init__(state_sparse_to_dense)
        self.action_sparse_to_dense = action_sparse_to_dense

    def preprocess(self, batch) -> rlt.RawTrainingBatch:
        training_batch = super().preprocess(batch)

        actions, actions_presence = self.action_sparse_to_dense(batch["action"])

        next_actions, next_actions_presence = self.action_sparse_to_dense(
            batch["next_action"]
        )

        max_action_size = max(len(pna) for pna in batch["possible_next_actions"])

        pnas_mask = torch.Tensor(
            [
                ([1] * len(l) + [0] * (max_action_size - len(l)))
                for l in batch["possible_next_actions"]
            ]
        ).byte()
        flat_pnas: List[Dict[int, float]] = []
        for pa in batch["possible_next_actions"]:
            flat_pnas.extend(pa)
            for _ in range(max_action_size - len(pa)):
                flat_pnas.append({})

        not_terminal = torch.any(next_actions_presence > 0, 1).float().reshape(-1, 1)
        pnas, pnas_presence = self.action_sparse_to_dense(flat_pnas)

        base_input = cast(rlt.RawBaseInput, training_batch.training_input)
        tiled_next_state = torch.repeat_interleave(
            base_input.next_state.float_features.value, max_action_size, dim=0
        )
        tiled_next_state_presence = torch.repeat_interleave(
            base_input.next_state.float_features.presence, max_action_size, dim=0
        )

        pas_mask = torch.Tensor(
            [
                ([1] * len(l) + [0] * (max_action_size - len(l)))
                for l in batch["possible_actions"]
            ]
        ).byte()
        flat_pas: List[Dict[int, float]] = []
        for pa in batch["possible_actions"]:
            flat_pas.extend(pa)
            for _ in range(max_action_size - len(pa)):
                flat_pas.append({})
        pas, pas_presence = self.action_sparse_to_dense(flat_pas)

        return training_batch._replace(
            training_input=rlt.RawParametricDqnInput(
                state=base_input.state,
                reward=base_input.reward,
                time_diff=base_input.time_diff,
                action=rlt.FeatureVector(
                    float_features=rlt.ValuePresence(
                        value=actions, presence=actions_presence
                    )
                ),
                next_action=rlt.FeatureVector(
                    float_features=rlt.ValuePresence(
                        value=next_actions, presence=next_actions_presence
                    )
                ),
                not_terminal=not_terminal,
                next_state=base_input.next_state,
                tiled_next_state=rlt.FeatureVector(
                    float_features=rlt.ValuePresence(
                        value=tiled_next_state, presence=tiled_next_state_presence
                    )
                ),
                possible_actions=rlt.FeatureVector(
                    float_features=rlt.ValuePresence(value=pas, presence=pas_presence)
                ),
                possible_actions_mask=pas_mask,
                possible_next_actions=rlt.FeatureVector(
                    float_features=rlt.ValuePresence(value=pnas, presence=pnas_presence)
                ),
                possible_next_actions_mask=pnas_mask,
                step=None,
            )
        )


class ContinuousPreprocessHandler(PreprocessHandler):
    def __init__(
        self,
        state_sparse_to_dense: SparseToDenseProcessor,
        action_sparse_to_dense: SparseToDenseProcessor,
    ):
        super().__init__(state_sparse_to_dense)
        self.action_sparse_to_dense = action_sparse_to_dense

    def preprocess(self, batch) -> rlt.RawTrainingBatch:
        training_batch = super().preprocess(batch)

        actions = self.action_sparse_to_dense(batch["action"])
        next_actions = self.action_sparse_to_dense(batch["next_action"])

        not_terminal = torch.from_numpy(
            np.array(batch["next_action"], dtype=np.bool).astype(np.float32)
        ).reshape(-1, 1)

        base_input = cast(rlt.RawBaseInput, training_batch.training_input)
        return training_batch._replace(
            training_input=rlt.RawPolicyNetworkInput(
                state=base_input.state,
                action=actions,
                next_state=base_input.next_state,
                next_action=next_actions,
                reward=base_input.reward,
                not_terminal=not_terminal,
                time_diff=base_input.time_diff,
                step=None,
            )
        )
