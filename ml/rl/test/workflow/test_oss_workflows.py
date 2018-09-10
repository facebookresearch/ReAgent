#!/usr/bin/env python3

import os
import unittest

from ml.rl.workflow.dqn_workflow import train_network


curr_dir = os.path.dirname(__file__)


class TestOSSWorkflows(unittest.TestCase):
    def test_dqn_workflow(self):
        """Run DQN workflow to ensure no crashes, algorithm correctness
        not tested here."""
        params = {
            "training_data_path": os.path.join(
                curr_dir, "test_data/cartpole_training_data.json"
            ),
            "state_norm_data_path": os.path.join(
                curr_dir, "test_data/cartpole_norm.json"
            ),
            "model_output_path": None,
            "use_gpu": False,
            "actions": ["0", "1"],
            "epochs": 1,
            "rl": {},
            "rainbow": {},
            "training": {"minibatch_size": 16},
        }
        predictor = train_network(params)
        test_float_state_features = [{"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0}]
        q_values = predictor.predict(test_float_state_features)
        assert len(q_values[0].keys()) == 2
