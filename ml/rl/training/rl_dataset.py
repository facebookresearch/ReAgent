#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import gzip
import json
import logging
import pickle

import numpy as np


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RLDataset:
    def __init__(self, file_path):
        """
        Holds a collection of RL samples:
            1) Can insert in the "pre-timeline" format (file extension json)
            2) Can insert in the "replay buffer" format (file extension pkl)

        :param file_path: String load/save the dataset from/to this file.
        """
        file_extension = file_path.split(".")[-1]
        assert file_extension in (
            "json",
            "pkl",
        ), "File type {} not supported. Only json and pkl supported."
        self.use_pickle = True if file_extension == "pkl" else False
        self.file_path = file_path
        self.rows = []

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
        if self.use_pickle:
            kwargs.pop("mdp_id", None)
            kwargs.pop("sequence_number", None)
            kwargs.pop("action_probability", None)
            kwargs.pop("timeline_format_action", None)
            self.insert_replay_buffer_format(**kwargs)
        else:
            kwargs.pop("next_state", None)
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
        timeline_format_action,
        reward,
        terminal,
        possible_actions,
        time_diff,
        action_probability,
        possible_actions_mask,
    ):
        """
        Insert a new sample to the dataset in the pre-timeline json format.
        Format needed for running timeline operator and for uploading dataset to hive.
        """
        state = state.tolist()
        action = timeline_format_action
        if possible_actions:
            possible_actions = possible_actions.tolist()
        else:
            possible_actions = possible_actions_mask

        assert isinstance(state, list)
        assert isinstance(action, (list, str))
        assert isinstance(reward, (float, int))
        assert isinstance(terminal, bool)
        assert possible_actions is None or isinstance(
            possible_actions, (list, np.ndarray)
        ), f"Expecting list/np.ndarray; got {type(possible_actions)}"
        assert isinstance(time_diff, int)
        assert isinstance(action_probability, float)

        state_features = {str(i): v for i, v in enumerate(state)}

        # This assumes that every state feature is present in every training example.
        int_state_feature_keys = [int(k) for k in state_features.keys()]
        idx_bump = max(int_state_feature_keys) + 1
        if isinstance(action, list):
            # Parametric or continuous action domain
            action = {str(k + idx_bump): v for k, v in enumerate(action)}
            for k, v in list(action.items()):
                if v == 0:
                    del action[k]

        if possible_actions is None:
            # Continuous action
            pass
        elif len(possible_actions) == 0:
            # Parametric action with no possible actions
            pass
        elif isinstance(possible_actions[0], int):
            # Discrete action domain
            possible_actions = [
                str(idx) for idx, val in enumerate(possible_actions) if val == 1
            ]
        elif isinstance(possible_actions[0], list):
            # Parametric action
            if terminal:
                possible_actions = []
            else:
                possible_actions = [
                    {str(k + idx_bump): v for k, v in enumerate(action)}
                    for action in possible_actions
                ]
                for a in possible_actions:
                    for k, v in list(a.items()):
                        if v == 0:
                            del a[k]
        else:
            print(possible_actions)
            raise NotImplementedError()

        self.rows.append(
            {
                "ds": "2019-01-01",  # Fix ds for simplicity in open source examples
                "mdp_id": str(mdp_id),
                "sequence_number": int(sequence_number),
                "state_features": state_features,
                "action": action,
                "reward": reward,
                "action_probability": action_probability,
                "possible_actions": possible_actions,
                "metrics": {"reward": reward},
            }
        )
