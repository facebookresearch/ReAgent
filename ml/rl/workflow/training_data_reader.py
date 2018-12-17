#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import bz2
import gzip
import itertools
import json
import os

import numpy as np
import pandas as pd
import torch
from ml.rl.preprocessing import normalization
from ml.rl.training.training_data_page import TrainingDataPage


class JSONDataset:
    """Create the reader for a JSON training dataset."""

    def __init__(self, path, batch_size=None, converter=None):
        self.path = os.path.expanduser(path)
        self.file_type = path.split(".")[-1]
        self.batch_size = batch_size
        self.len = self.line_count()
        self.reset_iterator()

    def reset_iterator(self):
        self.data_iterator = pd.read_json(
            self.path, lines=True, chunksize=self.batch_size
        )

    def read_batch(self, astype="dict"):
        assert (
            self.batch_size is not None
        ), "Batch size must be provided to read data in batches."

        try:
            x = next(self.data_iterator)
        except StopIteration:
            # No more data to read
            return None
        if astype == "dict":
            return x.to_dict(orient="list")
        return x

    def read_all(self):
        return pd.read_json(self.path, lines=True)
 
    def __len__(self):
        return self.len

    def line_count(self):
        lines = 0
        if self.file_type == "gz":
            with gzip.open(self.path) as f:
                for _ in f:
                    lines += 1
        elif self.file_type == "bz2":
            with bz2.open(self.path) as f:
                for _ in f:
                    lines += 1
        else:
            with open(self.path) as f:
                for _ in f:
                    lines += 1
        return lines


def read_norm_file(path):
    path = os.path.expanduser(path)
    if path.split(".")[-1] == "gz":
        with gzip.open(path) as f:
            norm_json = json.load(f)
    else:
        with open(path) as f:
            norm_json = json.load(f)
    return normalization.deserialize(norm_json)


def read_actions(action_names, actions):
    actions = np.array(actions, dtype=np.str)
    actions = np.expand_dims(actions, axis=1)
    action_names_tiled = np.tile(action_names, actions.shape)
    return torch.tensor(
        (actions == action_names_tiled).astype(np.int64), dtype=torch.int64
    )


def pandas_sparse_to_dense(feature_name_list, batch):
    state_features_df = pd.DataFrame(batch).fillna(normalization.MISSING_VALUE)
    # Add columns identified by normalization, but not present in batch
    for col in feature_name_list:
        if col not in state_features_df.columns:
            state_features_df[col] = normalization.MISSING_VALUE
    return torch.from_numpy(state_features_df[feature_name_list].values).float()


def preprocess_batch_for_training(
    state_preprocessor, batch, action_names=None, action_preprocessor=None
):

    assert (action_names is None) ^ (
        action_preprocessor is None
    ), "Either action_names should be None xor action_preprocessor should be None"

    # Preprocess state features
    sorted_state_features, _ = state_preprocessor._sort_features_by_normalization()
    sorted_state_features_str = [str(x) for x in sorted_state_features]
    state_features_dense = pandas_sparse_to_dense(
        sorted_state_features_str, batch["state_features"]
    )
    next_state_features_dense = pandas_sparse_to_dense(
        sorted_state_features_str, batch["next_state_features"]
    )

    state_features_dense = state_preprocessor.forward(state_features_dense)
    next_state_features_dense = state_preprocessor.forward(next_state_features_dense)

    mdp_ids = np.array(batch["mdp_id"]).reshape(-1, 1)
    sequence_numbers = torch.tensor(
        batch["sequence_number"], dtype=torch.int32
    ).reshape(-1, 1)
    rewards = torch.tensor(batch["reward"], dtype=torch.float32).reshape(-1, 1)
    time_diffs = torch.tensor(batch["time_diff"], dtype=torch.int32).reshape(-1, 1)

    if action_preprocessor:
        # Preprocess action features for parametric action DQN
        sorted_action_features, _ = (
            action_preprocessor._sort_features_by_normalization()
        )
        sorted_action_features_str = [str(x) for x in sorted_action_features]
        actions = pandas_sparse_to_dense(sorted_action_features_str, batch["action"])

        if "possible_next_actions" not in batch.keys():
            # DDPG / SAC
            not_terminal = torch.from_numpy(
                np.array(batch["next_action"], dtype=np.bool).astype(np.float32)
            ).reshape(-1, 1)
            pnas, pnas_mask, possible_next_state_actions = None, None, None
            pas, pas_mask, possible_state_actions = None, None, None
            next_actions = None
        else:
            # Parametric DQN
            actions = action_preprocessor.forward(actions)
            next_actions = pandas_sparse_to_dense(
                sorted_action_features_str, batch["next_action"]
            )
            next_actions = action_preprocessor.forward(next_actions)

            max_action_size = max(len(pna) for pna in batch["possible_next_actions"])

            pas_mask = torch.Tensor(
                [
                    ([1] * len(l) + [0] * (max_action_size - len(l)))
                    for l in batch["possible_actions"]
                ]
            )
            flat_pas = []
            for pa in enumerate(batch["possible_actions"]):
                for single_pa in enumerate(pa):
                    flat_pas.append(single_pa)
                for _ in range(max_action_size - len(pa)):
                    flat_pas.append({})

            pnas_mask = torch.Tensor(
                [
                    ([1] * len(l) + [0] * (max_action_size - len(l)))
                    for l in batch["possible_next_actions"]
                ]
            )
            flat_pnas = []
            for pna in enumerate(batch["possible_next_actions"]):
                for single_pna in enumerate(pna):
                    flat_pnas.append(single_pna)
                for _ in range(max_action_size - len(pna)):
                    flat_pnas.append({})

            not_terminal = torch.from_numpy(
                np.array(
                    [len(pna) > 0 for pna in batch["possible_next_actions"]]
                ).astype(np.float32)
            ).reshape(-1, 1)
            pnas = pandas_sparse_to_dense(sorted_action_features_str, flat_pnas)
            pnas = action_preprocessor.forward(pnas)
            tiled_next_state_features_dense = next_state_features_dense.repeat(
                1, max_action_size
            ).reshape(-1, next_state_features_dense.shape[1])

            possible_next_state_actions = torch.cat(
                (tiled_next_state_features_dense, pnas.cpu()), dim=1
            )

            pas_mask = torch.Tensor(
                [
                    ([1] * len(l) + [0] * (max_action_size - len(l)))
                    for l in batch["possible_actions"]
                ]
            )
            flat_pas = []
            for pa in batch["possible_actions"]:
                flat_pas.extend(pa)
                for _ in range(max_action_size - len(pa)):
                    flat_pas.append({})
            pas = pandas_sparse_to_dense(sorted_action_features_str, flat_pas)
            pas = action_preprocessor.forward(pas)

            tiled_state_features_dense = state_features_dense.repeat(
                1, max_action_size
            ).reshape(-1, state_features_dense.shape[1])

            possible_state_actions = torch.cat(
                (tiled_state_features_dense, pas.cpu()), dim=1
            )
    else:
        actions = read_actions(action_names, batch["action"])
        pas_mask = torch.from_numpy(
            np.array(batch["possible_actions"], dtype=np.float32)
        )

        next_actions = read_actions(action_names, batch["next_action"])
        pnas_mask = np.array(batch["possible_next_actions"], dtype=np.float32)
        not_terminal = torch.from_numpy(
            np.max(pnas_mask, 1).astype(np.float32).reshape(-1, 1)
        ).float()
        pnas_mask = torch.from_numpy(pnas_mask)

        pnas, possible_next_state_actions = None, None
        pas, possible_state_actions = None, None

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
        actions=actions,
        propensities=propensities,
        rewards=rewards,
        possible_actions_mask=pas_mask,
        next_states=next_state_features_dense,
        next_actions=next_actions,
        possible_next_actions_mask=pnas_mask,
        not_terminal=not_terminal,
        time_diffs=time_diffs,
        possible_actions_state_concat=possible_state_actions,
        possible_next_actions_state_concat=possible_next_state_actions,
    )
