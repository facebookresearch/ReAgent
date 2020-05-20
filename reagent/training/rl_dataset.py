#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import gzip
import json
import logging
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RLDataset:
    # TODO: delete file_path after mdnrnn test in old gym environment is migrated
    def __init__(self, file_path: Optional[str] = None):
        """
        Holds a collection of RL samples:
            1) Can insert in the "pre-timeline" format (file extension json)
            2) Can insert in the "replay buffer" format (file extension pkl)

        :param file_path: String load/save the dataset from/to this file.
        """
        file_extension = None
        if file_path is not None:
            file_extension = file_path.split(".")[-1]
            assert file_extension in (
                "json",
                "pkl",
            ), "File type {} not supported. Only json and pkl supported."
            self.file_path = file_path

        self.use_pickle = True if file_extension == "pkl" else False
        self.rows: List[Dict] = []

    def to_pandas_df(self):
        return pd.DataFrame(self.rows)

    def __len__(self):
        return len(self.rows)

    def load(self):
        """Load samples from a gzipped json file."""
        with gzip.open(self.file_path) as f:
            for line in f:
                self.rows.append(json.loads(line))

    def save(self):
        """Save samples as a pickle file or JSON file."""
        if self.use_pickle:
            with open(self.file_path, "wb") as f:
                pickle.dump(self.rows, f)
        else:
            with open(self.file_path, "w") as f:
                for data in self.rows:
                    json.dump(data, f)
                    f.write("\n")
        logger.info("RLDataset saved to {}".format(self.file_path))

    def insert(self, **kwargs):
        # we do not see any use case with time_diff != 1
        assert kwargs["time_diff"] == 1

        if self.use_pickle:
            kwargs.pop("mdp_id", None)
            kwargs.pop("sequence_number", None)
            kwargs.pop("action_probability", None)
            kwargs.pop("timeline_format_action", None)
            self.insert_replay_buffer_format(**kwargs)
            return
        else:
            kwargs.pop("next_state", None)
            kwargs.pop("terminal", None)
            kwargs.pop("next_action", None)
            kwargs.pop("action", None)
            kwargs.pop("possible_next_actions", None)
            kwargs.pop("possible_next_actions_mask", None)
            kwargs.pop("policy_id", None)
            self.insert_pre_timeline_format(**kwargs)

    def insert_replay_buffer_format(
        self,
        state,
        action,
        reward,
        next_state,
        next_action,
        terminal,
        possible_next_actions,
        possible_next_actions_mask,
        time_diff,
        possible_actions,
        possible_actions_mask,
        policy_id,
    ):
        """
        Insert a new sample to the dataset in the same format as the
        replay buffer.
        """
        self.rows.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "next_action": next_action,
                "terminal": terminal,
                "possible_next_actions": possible_next_actions,
                "possible_next_actions_mask": possible_next_actions_mask,
                "time_diff": time_diff,
                "next_action": next_action,
                "possible_actions": possible_actions,
                "possible_actions_mask": possible_actions_mask,
                "policy_id": policy_id,
            }
        )

    def insert_pre_timeline_format(
        self,
        mdp_id,
        sequence_number,
        state,
        action,
        reward,
        possible_actions,
        time_diff,
        action_probability,
        possible_actions_mask,
    ):
        """
        Insert a new sample to the dataset in the pre-timeline json format.
        Format needed for running timeline operator and for uploading dataset to hive.
        """

        assert isinstance(reward, float)
        assert isinstance(action_probability, float)
        row = {
            "ds": "2019-01-01",  # Fix ds for simplicity in open source examples
            "state_features": {i: v for i, v in enumerate(state.tolist())},
            "mdp_id": str(mdp_id),
            "sequence_number": int(sequence_number),
            "action_probability": action_probability,
            "reward": reward,
            "metrics": {"reward": reward},
        }

        # Discrete actions should be str
        # Parametric or continuou actions should be str -> double
        assert isinstance(action, (str, dict))
        row["action"] = action

        # if both None, we assume continuous action domain, and ignore possible_actions
        possible_actions = (
            possible_actions_mask if possible_actions is None else possible_actions
        )
        if possible_actions is not None:
            if isinstance(possible_actions, torch.Tensor) and isinstance(
                possible_actions[0].item(), int
            ):
                # Discrete action domain
                possible_actions = [
                    str(idx) for idx, val in enumerate(possible_actions) if val == 1
                ]
            elif isinstance(possible_actions[0], list):
                # Parametric action
                possible_actions = [
                    {str(k): v for k, v in enumerate(action) if v != 0}
                    for action in possible_actions
                ]
            else:
                raise NotImplementedError(
                    f"Got {type(possible_actions)}, value: {possible_actions}"
                )
            row["possible_actions"] = possible_actions

        self.rows.append(row)
