#!/usr/bin/env python3

import json
import os

import numpy as np
import pandas as pd
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


def preprocess_batch_for_training(preprocessor, action_names, batch):
    sorted_features, _ = preprocessor._sort_features_by_normalization()
    sorted_features_str = [str(x) for x in sorted_features]
    state_features_dense = pandas_sparse_to_dense(
        sorted_features_str, batch["state_features"]
    )
    next_state_features_dense = pandas_sparse_to_dense(
        sorted_features_str, batch["next_state_features"]
    )
    mdp_ids = np.array(batch["mdp_id"])
    sequence_numbers = np.array(batch["sequence_number"], dtype=np.int32)
    actions = read_actions(action_names, batch["action"])
    pnas = np.array(batch["possible_next_actions"], dtype=np.float32)
    rewards = np.array(batch["reward"], dtype=np.float32)
    time_diffs = np.array(batch["time_diff"], dtype=np.int32)
    not_terminals = np.max(pnas, 1).astype(np.bool)
    episode_values = np.array(batch["episode_value"], dtype=np.float32)
    if "action_probability" in batch:
        propensities = np.array(batch["action_probability"], dtype=np.float32)
    else:
        propensities = np.ones(shape=rewards.shape, dtype=np.float32)

    # Preprocess state features
    state_features_dense = preprocessor(state_features_dense)
    next_state_features_dense = preprocessor(next_state_features_dense)

    return TrainingDataPage(
        mdp_ids=mdp_ids,
        sequence_numbers=sequence_numbers,
        states=state_features_dense,
        actions=actions,
        propensities=propensities,
        rewards=rewards,
        next_states=next_state_features_dense,
        possible_next_actions=pnas,
        episode_values=episode_values,
        not_terminals=not_terminals,
        time_diffs=time_diffs,
    )
