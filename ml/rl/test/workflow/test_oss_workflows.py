#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import tempfile
import unittest

import torch
from ml.rl.tensorboardX import SummaryWriterContext
from ml.rl.training.dqn_predictor import DQNPredictor
from ml.rl.workflow import ddpg_workflow, dqn_workflow, parametric_dqn_workflow


curr_dir = os.path.dirname(__file__)


class TestOSSWorkflows(unittest.TestCase):
    def setUp(self):
        SummaryWriterContext._reset_globals()

    def tearDown(self):
        SummaryWriterContext._reset_globals()

    def _test_dqn_workflow(self, use_gpu=False, use_all_avail_gpus=False):
        # TODO: Re-enable this test in D13445814
        return
        """Run DQN workflow to ensure no crashes, algorithm correctness
        not tested here."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            params = {
                "training_data_path": os.path.join(
                    curr_dir, "test_data/discrete_action/cartpole_training_data.json"
                ),
                "eval_data_path": os.path.join(
                    curr_dir, "test_data/discrete_action/cartpole_eval_data.json"
                ),
                "state_norm_data_path": os.path.join(
                    curr_dir, "test_data/discrete_action/cartpole_norm.json"
                ),
                "model_output_path": tmpdirname,
                "use_gpu": use_gpu,
                "use_all_avail_gpus": use_all_avail_gpus,
                "actions": ["0", "1"],
                "epochs": 5,
                "rl": {},
                "rainbow": {},
                "training": {"minibatch_size": 16},
                "in_training_cpe": {"mdp_sampled_rate": 1.0},
            }
            predictor = dqn_workflow.train_network(params)
            test_float_state_features = [{"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0}]
            q_values = predictor.predict(test_float_state_features)
        assert len(q_values[0].keys()) == 2

    def test_dqn_workflow(self):
        self._test_dqn_workflow()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_dqn_workflow_gpu(self):
        self._test_dqn_workflow(use_gpu=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_dqn_workflow_all_gpus(self):
        self._test_dqn_workflow(use_gpu=True, use_all_avail_gpus=True)

    def _test_parametric_dqn_workflow(self, use_gpu=False, use_all_avail_gpus=False):
        # TODO: Re-enable this test in D13445814
        return
        """Run Parametric DQN workflow to ensure no crashes, algorithm correctness
        not tested here."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            params = {
                "training_data_path": os.path.join(
                    curr_dir, "test_data/parametric_action/cartpole_training_data.json"
                ),
                "eval_data_path": os.path.join(
                    curr_dir, "test_data/parametric_action/cartpole_eval_data.json"
                ),
                "state_norm_data_path": os.path.join(
                    curr_dir, "test_data/parametric_action/state_features_norm.json"
                ),
                "action_norm_data_path": os.path.join(
                    curr_dir, "test_data/parametric_action/action_norm.json"
                ),
                "model_output_path": tmpdirname,
                "use_gpu": use_gpu,
                "use_all_avail_gpus": use_all_avail_gpus,
                "epochs": 5,
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

    def test_parametric_dqn_workflow(self):
        self._test_parametric_dqn_workflow()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_parametric_dqn_workflow_gpu(self):
        self._test_parametric_dqn_workflow(use_gpu=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_parametric_dqn_workflow_all_gpus(self):
        self._test_parametric_dqn_workflow(use_gpu=True, use_all_avail_gpus=True)

    def _test_ddpg_workflow(self, use_gpu=False, use_all_avail_gpus=False):
        # TODO: Re-enable this test in D13445814
        return
        """Run DDPG workflow to ensure no crashes, algorithm correctness
        not tested here."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            params = {
                "training_data_path": os.path.join(
                    curr_dir, "test_data/continuous_action/pendulum_training_data.json"
                ),
                "eval_data_path": os.path.join(
                    curr_dir, "test_data/continuous_action/pendulum_eval_data.json"
                ),
                "state_norm_data_path": os.path.join(
                    curr_dir, "test_data/continuous_action/state_features_norm.json"
                ),
                "action_norm_data_path": os.path.join(
                    curr_dir, "test_data/continuous_action/action_norm.json"
                ),
                "model_output_path": tmpdirname,
                "use_gpu": use_gpu,
                "use_all_avail_gpus": use_all_avail_gpus,
                "epochs": 5,
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

    def test_ddpg_workflow(self):
        self._test_ddpg_workflow()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ddpg_workflow_gpu(self):
        self._test_ddpg_workflow(use_gpu=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ddpg_workflow_all_gpus(self):
        self._test_ddpg_workflow(use_gpu=True, use_all_avail_gpus=True)

    def test_read_c2_model_from_file(self):
        """Test reading output caffe2 model from file and using it for inference."""
        path = os.path.join(curr_dir, "test_data/discrete_action/example_predictor.c2")
        predictor = DQNPredictor.load(path, "minidb", int_features=False)
        test_float_state_features = [{"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0}]
        q_values = predictor.predict(test_float_state_features)
        assert len(q_values[0].keys()) == 2
