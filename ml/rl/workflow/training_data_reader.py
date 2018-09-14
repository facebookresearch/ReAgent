#!/usr/bin/env python3

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
        self.read_data_in_chunks = self._get_chunked_reading_setting()
        self.batch_size = batch_size
        self.len = self.line_count()
        if self.read_data_in_chunks:
            # Do not read entire dataset into memory
            self.column_names = next(
                pd.read_json(self.path, lines=True, chunksize=1)
            ).columns.values
        else:
            # Read entire dataset into memory
            self.data = pd.read_json(self.path, lines=True)
            self.column_names = self.data.columns.values

    def _get_chunked_reading_setting(self):
        """FB internal is currently pinned to Pandas 0.20.3 which does
        not support chunked reading from JSON files. This is a hacky
        work around to enable chunked reading for open source."""
        self._pd_version_int = int("".join(pd.__version__.split(".")))
        self._min_pd_version_for_chunking = 210
        return self._pd_version_int >= self._min_pd_version_for_chunking

    def read_batch(self, index, astype="dict"):
        assert (
            self.batch_size is not None
        ), "Batch size must be provided to read data in batches."

        starting_row = index * self.batch_size
        ending_row = starting_row + self.batch_size
        if self.read_data_in_chunks:
            x = next(
                pd.read_json(
                    self.path,
                    lines=True,
                    skiprows=starting_row + 1,  # +1 to skip the header
                    chunksize=self.batch_size,
                    names=self.column_names,
                )
            )
            if astype == "dict":
                return x.to_dict(orient="list")
            return x
        else:
            if astype == "dict":
                return self.data[starting_row:ending_row].to_dict(orient="list")
            return self.data[starting_row:ending_row]

    def read_all(self):
        if self.read_data_in_chunks:
            return pd.read_json(self.path, lines=True)
        else:
            return self.data

    def __len__(self):
        return self.len

    def line_count(self):
        lines = 0
        with open(self.path) as f:
            for _ in f:
                lines += 1
        return lines


def read_norm_file(path):
    path = os.path.expanduser(path)
    with open(path) as f:
        norm_json = json.load(f)
    return normalization.deserialize(norm_json)


def read_actions(action_names, actions):
    actions = np.array(actions, dtype=np.str)
    actions = np.expand_dims(actions, axis=1)
    action_names_tiled = np.tile(action_names, actions.shape)
    return (actions == action_names_tiled).astype(int)


def pandas_sparse_to_dense(feature_name_list, batch):
    state_features_df = pd.DataFrame(batch).fillna(normalization.MISSING_VALUE)
    # Add columns identified by normalization, but not present in batch
    for col in feature_name_list:
        if col not in state_features_df.columns:
            state_features_df[col] = normalization.MISSING_VALUE
    return state_features_df[feature_name_list].values


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

    mdp_ids = np.array(batch["mdp_id"])
    sequence_numbers = np.array(batch["sequence_number"], dtype=np.int32)
    rewards = np.array(batch["reward"], dtype=np.float32)
    time_diffs = np.array(batch["time_diff"], dtype=np.int32)
    episode_values = np.array(batch["episode_value"], dtype=np.float32)

    if action_preprocessor:
        # Preprocess action features for parametric action DQN
        sorted_action_features, _ = (
            action_preprocessor._sort_features_by_normalization()
        )
        sorted_action_features_str = [str(x) for x in sorted_action_features]
        actions = pandas_sparse_to_dense(sorted_action_features_str, batch["action"])
        pnas_lens = np.array([len(l) for l in batch["possible_next_actions"]])
        flat_pnas = list(itertools.chain.from_iterable(batch["possible_next_actions"]))
        not_terminals = pnas_lens.astype(np.bool)
        pnas = pandas_sparse_to_dense(sorted_action_features_str, flat_pnas)
        actions = action_preprocessor.forward(actions)
        pnas = action_preprocessor.forward(pnas)
        tiled_next_state_features_dense = np.repeat(
            next_state_features_dense, pnas_lens, axis=0
        )
        possible_next_state_actions = torch.cat(
            (tiled_next_state_features_dense, pnas), dim=1
        )
        pas_lens = np.array([len(l) for l in batch["possible_actions"]])
        flat_pas = list(itertools.chain.from_iterable(batch["possible_actions"]))
        pas = pandas_sparse_to_dense(sorted_action_features_str, flat_pas)
        pas = action_preprocessor.forward(pas)
        tiled_state_features_dense = np.repeat(state_features_dense, pas_lens, axis=0)
        possible_state_actions = torch.cat((tiled_state_features_dense, pas), dim=1)
    else:
        actions = read_actions(action_names, batch["action"])
        pnas = np.array(batch["possible_next_actions"], dtype=np.float32)
        not_terminals = np.max(pnas, 1).astype(np.bool)
        pnas_lens, possible_next_state_actions = None, None
        pas, pas_lens, possible_state_actions = None, None, None
    if "propensity" in batch:
        propensities = np.array(batch["propensity"], dtype=np.float32)
    else:
        propensities = np.ones(shape=rewards.shape, dtype=np.float32)

    return TrainingDataPage(
        mdp_ids=mdp_ids,
        sequence_numbers=sequence_numbers,
        states=state_features_dense,
        actions=actions,
        propensities=propensities,
        rewards=rewards,
        possible_actions=pas,
        possible_actions_lengths=pas_lens,
        next_states=next_state_features_dense,
        possible_next_actions=pnas,
        possible_next_actions_lengths=pnas_lens,
        episode_values=episode_values,
        not_terminals=not_terminals,
        time_diffs=time_diffs,
        state_pas_concat=possible_state_actions,
        next_state_pnas_concat=possible_next_state_actions,
    )
