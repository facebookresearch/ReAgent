#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import gzip
import json
from datetime import datetime

import numpy as np


class RLDataset:
    def __init__(self, file_path):
        """
        Holds a collection of RL samples in the "pre-timeline" format.

        :param file_path: String Load/save the dataset from/to this file.
        """
        self.file_path = file_path
        self.rows = []

    def load(self):
        """Load samples from a gzipped json file."""
        with gzip.open(self.file_path) as f:
            self.rows = json.load(f)

    def save(self):
        """Save samples as a JSON file."""
        data = self.rows

        with open(self.file_path, "w") as f:
            json.dump(data, f)

    def insert(
        self,
        mdp_id,
        sequence_number,
        state,
        action,
        reward,
        terminal,
        possible_actions,
        time_diff,
        action_probability,
    ):
        """
        Insert a new sample to the dataset.
        """

        assert isinstance(state, list)
        assert isinstance(action, (list, str))
        assert isinstance(reward, float)
        assert isinstance(terminal, bool)
        assert possible_actions is None or isinstance(
            possible_actions, (list, np.ndarray)
        )
        assert isinstance(time_diff, int)
        assert isinstance(action_probability, float)

        state_features = {str(i): v for i, v in enumerate(state)}

        # This assumes that every state feature is present in every training example.
        int_state_feature_keys = [int(k) for k in state_features.keys()]
        idx_bump = max(int_state_feature_keys) + 1
        if isinstance(action, list):
            # Parametric or continuous action domain
            action = {str(k + idx_bump): v for k, v in enumerate(action)}
            for k,v in list(action.items()):
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
                for k,v in list(a.items()):
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
