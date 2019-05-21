#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Dict, List

import numpy as np
import torch
from ml.rl.preprocessing.sparse_to_dense import SparseToDenseProcessor
from ml.rl.training.training_data_page import TrainingDataPage


class PreprocessHandler:
    def __init__(
        self, state_preprocessor, sparse_to_dense_processor: SparseToDenseProcessor
    ):
        self.state_preprocessor = state_preprocessor
        self.sparse_to_dense_processor = sparse_to_dense_processor

    def preprocess(self, batch) -> TrainingDataPage:
        # Preprocess state features
        sorted_state_features, _ = (
            self.state_preprocessor._sort_features_by_normalization()
        )
        sorted_state_features_str = [str(x) for x in sorted_state_features]
        state_features_dense = self.sparse_to_dense_processor(
            sorted_state_features_str, batch["state_features"]
        )
        next_state_features_dense = self.sparse_to_dense_processor(
            sorted_state_features_str, batch["next_state_features"]
        )

        state_features_dense = self.state_preprocessor.forward(state_features_dense)
        next_state_features_dense = self.state_preprocessor.forward(
            next_state_features_dense
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

        return TrainingDataPage(
            mdp_ids=mdp_ids,
            sequence_numbers=sequence_numbers,
            states=state_features_dense,
            propensities=propensities,
            rewards=rewards,
            next_states=next_state_features_dense,
            time_diffs=time_diffs,
        )


class DqnPreprocessHandler(PreprocessHandler):
    def __init__(
        self,
        state_preprocessor,
        action_names: List[str],
        sparse_to_dense_processor: SparseToDenseProcessor,
    ):
        super().__init__(state_preprocessor, sparse_to_dense_processor)
        self.action_names = action_names

    def read_actions(self, actions):
        actions = np.array(actions, dtype=np.str)
        actions = np.expand_dims(actions, axis=1)
        action_names_tiled = np.tile(self.action_names, actions.shape)
        return torch.tensor(
            (actions == action_names_tiled).astype(np.int64), dtype=torch.int64
        )

    def preprocess(self, batch) -> TrainingDataPage:
        tdp = super().preprocess(batch)
        actions = self.read_actions(batch["action"])
        pas_mask = torch.from_numpy(
            np.array(batch["possible_actions"], dtype=np.float32)
        )

        next_actions = self.read_actions(batch["next_action"])
        pnas_mask = np.array(batch["possible_next_actions"], dtype=np.float32)
        not_terminal = torch.from_numpy(
            np.max(pnas_mask, 1).astype(np.float32).reshape(-1, 1)
        ).float()
        pnas_mask = torch.from_numpy(pnas_mask)

        possible_next_state_actions = None
        possible_state_actions = None

        return TrainingDataPage(
            mdp_ids=tdp.mdp_ids,
            sequence_numbers=tdp.sequence_numbers,
            states=tdp.states,
            actions=actions,
            propensities=tdp.propensities,
            rewards=tdp.rewards,
            possible_actions_mask=pas_mask,
            next_states=tdp.next_states,
            next_actions=next_actions,
            possible_next_actions_mask=pnas_mask,
            not_terminal=not_terminal,
            time_diffs=tdp.time_diffs,
            possible_actions_state_concat=possible_state_actions,
            possible_next_actions_state_concat=possible_next_state_actions,
        )


class ParametricDqnPreprocessHandler(PreprocessHandler):
    def __init__(
        self,
        state_preprocessor,
        action_preprocessor,
        sparse_to_dense_processor: SparseToDenseProcessor,
    ):
        super().__init__(state_preprocessor, sparse_to_dense_processor)
        self.action_preprocessor = action_preprocessor

    def preprocess(self, batch) -> TrainingDataPage:
        tdp = super().preprocess(batch)

        # Preprocess action features for parametric action DQN
        sorted_action_features, _ = (
            self.action_preprocessor._sort_features_by_normalization()
        )
        sorted_action_features_str = [str(x) for x in sorted_action_features]
        actions = self.sparse_to_dense_processor(
            sorted_action_features_str, batch["action"]
        )

        actions = self.action_preprocessor.forward(actions)
        next_actions = self.sparse_to_dense_processor(
            sorted_action_features_str, batch["next_action"]
        )
        next_actions = self.action_preprocessor.forward(next_actions)

        max_action_size = max(len(pna) for pna in batch["possible_next_actions"])

        pas_mask = torch.Tensor(
            [
                ([1] * len(l) + [0] * (max_action_size - len(l)))
                for l in batch["possible_actions"]
            ]
        )

        pnas_mask = torch.Tensor(
            [
                ([1] * len(l) + [0] * (max_action_size - len(l)))
                for l in batch["possible_next_actions"]
            ]
        )
        flat_pnas: List[Dict[int, float]] = []
        for pa in batch["possible_next_actions"]:
            flat_pnas.extend(pa)
            for _ in range(max_action_size - len(pa)):
                flat_pnas.append({})

        not_terminal = torch.from_numpy(
            np.array([len(pna) > 0 for pna in batch["possible_next_actions"]]).astype(
                np.float32
            )
        ).reshape(-1, 1)
        pnas = self.sparse_to_dense_processor(sorted_action_features_str, flat_pnas)
        pnas = self.action_preprocessor.forward(pnas)
        tiled_next_state_features_dense = tdp.next_states.repeat(
            1, max_action_size
        ).reshape(-1, tdp.next_states.shape[1])

        possible_next_state_actions = torch.cat(
            (tiled_next_state_features_dense, pnas.cpu()), dim=1
        )

        pas_mask = torch.Tensor(
            [
                ([1] * len(l) + [0] * (max_action_size - len(l)))
                for l in batch["possible_actions"]
            ]
        )
        flat_pas: List[Dict[int, float]] = []
        for pa in batch["possible_actions"]:
            flat_pas.extend(pa)
            for _ in range(max_action_size - len(pa)):
                flat_pas.append({})
        pas = self.sparse_to_dense_processor(sorted_action_features_str, flat_pas)
        pas = self.action_preprocessor.forward(pas)

        tiled_state_features_dense = tdp.states.repeat(1, max_action_size).reshape(
            -1, tdp.states.shape[1]
        )

        possible_state_actions = torch.cat(
            (tiled_state_features_dense, pas.cpu()), dim=1
        )

        return TrainingDataPage(
            mdp_ids=tdp.mdp_ids,
            sequence_numbers=tdp.sequence_numbers,
            states=tdp.states,
            actions=actions,
            propensities=tdp.propensities,
            rewards=tdp.rewards,
            possible_actions_mask=pas_mask,
            next_states=tdp.next_states,
            next_actions=next_actions,
            possible_next_actions_mask=pnas_mask,
            not_terminal=not_terminal,
            time_diffs=tdp.time_diffs,
            possible_actions_state_concat=possible_state_actions,
            possible_next_actions_state_concat=possible_next_state_actions,
            max_num_actions=max_action_size,
        )


class ContinuousPreprocessHandler(PreprocessHandler):
    def __init__(
        self,
        state_preprocessor,
        action_preprocessor,
        sparse_to_dense_processor: SparseToDenseProcessor,
    ):
        super().__init__(state_preprocessor, sparse_to_dense_processor)
        self.action_preprocessor = action_preprocessor

    def preprocess(self, batch) -> TrainingDataPage:
        tdp = super().preprocess(batch)

        sorted_action_features, _ = (
            self.action_preprocessor._sort_features_by_normalization()
        )
        sorted_action_features_str = [str(x) for x in sorted_action_features]
        actions = self.sparse_to_dense_processor(
            sorted_action_features_str, batch["action"]
        )

        not_terminal = torch.from_numpy(
            np.array(batch["next_action"], dtype=np.bool).astype(np.float32)
        ).reshape(-1, 1)
        pnas_mask, possible_next_state_actions = None, None
        pas_mask, possible_state_actions = None, None
        next_actions = None

        return TrainingDataPage(
            mdp_ids=tdp.mdp_ids,
            sequence_numbers=tdp.sequence_numbers,
            states=tdp.states,
            actions=actions,
            propensities=tdp.propensities,
            rewards=tdp.rewards,
            possible_actions_mask=pas_mask,
            next_states=tdp.next_states,
            next_actions=next_actions,
            possible_next_actions_mask=pnas_mask,
            not_terminal=not_terminal,
            time_diffs=tdp.time_diffs,
            possible_actions_state_concat=possible_state_actions,
            possible_next_actions_state_concat=possible_next_state_actions,
        )
