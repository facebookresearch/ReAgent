#!/usr/bin/env python3

import os
import unittest

from ml.rl.workflow import ddpg_workflow, dqn_workflow, parametric_dqn_workflow


curr_dir = os.path.dirname(__file__)


class TestOSSWorkflows(unittest.TestCase):
    def test_dqn_workflow(self):
        """Run DQN workflow to ensure no crashes, algorithm correctness
        not tested here."""
        params = {
            "training_data_path": os.path.join(
                curr_dir, "test_data/discrete_action/cartpole_training_data.json"
            ),
            "state_norm_data_path": os.path.join(
                curr_dir, "test_data/discrete_action/cartpole_norm.json"
            ),
            "model_output_path": None,
            "use_gpu": False,
            "actions": ["0", "1"],
            "epochs": 1,
            "rl": {},
            "rainbow": {},
            "training": {"minibatch_size": 16},
            "in_training_cpe": None,
        }
        predictor = dqn_workflow.train_network(params)
        test_float_state_features = [{"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0}]
        q_values = predictor.predict(test_float_state_features)
        assert len(q_values[0].keys()) == 2

    def test_parametric_dqn_workflow(self):
        """Run Parametric DQN workflow to ensure no crashes, algorithm correctness
        not tested here."""
        params = {
            "training_data_path": os.path.join(
                curr_dir, "test_data/parametric_action/cartpole_training_data.json"
            ),
            "state_norm_data_path": os.path.join(
                curr_dir, "test_data/parametric_action/state_features_norm.json"
            ),
            "action_norm_data_path": os.path.join(
                curr_dir, "test_data/parametric_action/action_norm.json"
            ),
            "model_output_path": None,
            "use_gpu": False,
            "epochs": 1,
            "rl": {},
            "rainbow": {},
            "training": {"minibatch_size": 16},
            "in_training_cpe": None,
        }
        predictor = parametric_dqn_workflow.train_network(params)
        test_float_state_features = [{"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0}]
        test_int_state_features = [{}]
        test_action_features = [{"4": 0.0, "5": 1.0}]
        q_values = predictor.predict(
            test_float_state_features, test_int_state_features, test_action_features
        )
        assert len(q_values[0].keys()) == 1

    def test_ddpg_workflow(self):
        """Run DDPG workflow to ensure no crashes, algorithm correctness
        not tested here."""
        params = {
            "training_data_path": os.path.join(
                curr_dir, "test_data/continuous_action/pendulum_training_data.json"
            ),
            "state_norm_data_path": os.path.join(
                curr_dir, "test_data/continuous_action/state_features_norm.json"
            ),
            "action_norm_data_path": os.path.join(
                curr_dir, "test_data/continuous_action/action_norm.json"
            ),
            "model_output_path": None,
            "use_gpu": False,
            "epochs": 1,
            "rl": {},
            "rainbow": {},
            "shared_training": {"minibatch_size": 16},
            "actor_training": {},
            "critic_training": {},
        }
        predictor = ddpg_workflow.train_network(params)
        test_float_state_features = [{"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0}]
        test_int_state_features = [{}]
        action = predictor.actor_prediction(
            test_float_state_features, test_int_state_features
        )
        assert len(action) == 1
