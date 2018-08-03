#!/usr/bin/env python3

import gzip
import json

import numpy as np


class RLDataset:
    def __init__(self, file_path):
        """
        Holds a collection of RL samples.

        :param file_path: String Load/save the dataset from/to this file.
        """
        self.file_path = file_path
        self.rows = []
        self.replay_memory = []

    def load(self):
        """Load samples from a gzipped json file."""
        with gzip.open(self.file_path) as f:
            self.rows = json.load(f)
            self._format_for_replay_memory()

    def save(self):
        """Save samples as a JSON file."""
        with open(self.file_path, "w") as f:
            json.dump(self.rows, f)

    def insert(
        self,
        state,
        action,
        reward,
        next_state,
        next_action,
        terminal,
        possible_next_actions,
        possible_next_actions_lengths,
        time_diff,
    ):
        """
        Insert a new sample to the dataset.
        """

        assert isinstance(state, list)
        assert isinstance(action, list)
        assert isinstance(reward, float)
        assert isinstance(next_state, list)
        assert isinstance(next_action, list)
        assert isinstance(terminal, bool)
        assert possible_next_actions is None or isinstance(possible_next_actions, list)
        assert isinstance(possible_next_actions_lengths, int)
        assert isinstance(time_diff, int)

        self.rows.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "next_action": next_action,
                "terminal": terminal,
                "possible_next_actions": possible_next_actions,
                "possible_next_actions_lengths": possible_next_actions_lengths,
                "time_diff": time_diff,
            }
        )

    def _format_for_replay_memory(self):
        for row in self.rows:
            item = (
                np.float32(row["state"]),
                np.float32(row["action"]),
                np.float32(row["reward"]),
                np.float32(row["next_state"]),
                np.float32(row["next_action"]),
                row["terminal"],
                row["possible_next_actions"],
                row["possible_next_actions_lengths"],
                row["time_diff"],
            )
            self.replay_memory.append(item)
