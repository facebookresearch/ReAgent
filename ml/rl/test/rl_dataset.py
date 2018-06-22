#!/usr/bin/env python3

import json


class RLDataset:

    def __init__(self, file_path):
        """
        Holds a collection of RL samples.

        :param file_path: String Load/save the dataset from/to this file.
        """
        self.file_path = file_path
        self.rows = []

    def load(self):
        """Load samples from a JSON file."""
        with open(self.file_path) as f:
            self.rows = json.load(f)
        return self

    def save(self):
        """Save samples as a JSON file."""
        with open(self.file_path, 'w') as f:
            json.dump(self.rows, f)
        return self

    def insert(self, mdp_id, sequence_number, state, action, reward,
               possible_actions):
        """
        Insert a new sample to the dataset.
        """
        assert isinstance(state, list)
        assert isinstance(sequence_number, int)
        assert action is None or isinstance(action, str)
        assert isinstance(reward, float)
        assert isinstance(possible_actions, list)
        self.rows.append({
            'ds': 'None',
            'mdp_id': str(mdp_id),
            'sequence_number': sequence_number,
            'state_features': {str(i): v for i, v in enumerate(state)},
            'action': action,
            'reward': reward,
            'possible_actions': possible_actions,
        })
