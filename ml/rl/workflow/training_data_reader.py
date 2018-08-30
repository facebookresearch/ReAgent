#!/usr/bin/env python3

import numpy as np
import pandas as pd
from ml.rl.preprocessing import normalization, preprocessor_net
from ml.rl.training.training_data_page import TrainingDataPage


class JSONDataset:
    """Create the reader for a JSON training dataset."""

    def __init__(self, path, batch_size=None, converter=None):
        self.path = path
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

    def read_batch(self, index):
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
            return x.to_dict(orient="list")
        else:
            return self.data[starting_row:ending_row].to_dict(orient="list")

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


def read_norm_params(table_output):
    norm_data = dict(zip(table_output["feature"], table_output["normalization"]))
    return normalization.deserialize(norm_data)


def read_actions(action_names, actions):
    actions = np.array(actions, dtype=np.str)
    actions = np.expand_dims(actions, axis=1)
    action_names_tiled = np.tile(action_names, actions.shape)
    return (actions == action_names_tiled).astype(int)


def preprocess_batch_for_training(action_names, batch, state_normalization):
    sorted_features, _ = preprocessor_net.sort_features_by_normalization(
        state_normalization
    )
    sorted_features_str = [str(x) for x in sorted_features]

    state_features_df = pd.DataFrame(batch["state_features"])
    state_features_dense = state_features_df[sorted_features_str].values
    next_state_features_df = pd.DataFrame(batch["next_state_features"])
    next_state_features_dense = next_state_features_df[sorted_features_str].values
    actions = read_actions(action_names, batch["action"])
    pnas = np.array(batch["possible_next_actions"], dtype=np.float32)
    rewards = np.array(batch["reward"], dtype=np.float32)
    time_diffs = np.array(batch["time_diff"], dtype=np.int32)
    not_terminals = np.max(pnas, 1).astype(np.bool)
    episode_values = np.array(batch["episode_value"], dtype=np.float32)

    # Add preprocessing steps in PyTorch here

    return TrainingDataPage(
        states=state_features_dense,
        actions=actions,
        rewards=rewards,
        next_states=next_state_features_dense,
        possible_next_actions=pnas,
        episode_values=episode_values,
        not_terminals=not_terminals,
        time_diffs=time_diffs,
    )
