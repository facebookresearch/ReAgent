#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import glob
import os
import tempfile
import unittest
from pathlib import Path

import torch
from ml.rl.prediction.dqn_torch_predictor import (
    DiscreteDqnTorchPredictor,
    ParametricDqnTorchPredictor,
)
from ml.rl.test.base.horizon_test_base import HorizonTestBase
from ml.rl.training.dqn_predictor import DQNPredictor
from ml.rl.training.parametric_dqn_predictor import ParametricDQNPredictor
from ml.rl.workflow import dqn_workflow, parametric_dqn_workflow


curr_dir = os.path.dirname(__file__)


class TestOSSWorkflows(HorizonTestBase):
    def _test_dqn_workflow(self, use_gpu=False, use_all_avail_gpus=False):
        """Run DQN workflow to ensure no crashes, algorithm correctness
        not tested here."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            lockfile = os.path.join(tmpdirname, "multiprocess_lock")
            Path(lockfile).touch()
            params = {
                "training_data_path": os.path.join(
                    curr_dir, "test_data/discrete_action/cartpole_training.json.bz2"
                ),
                "eval_data_path": os.path.join(
                    curr_dir, "test_data/discrete_action/cartpole_eval.json.bz2"
                ),
                "state_norm_data_path": os.path.join(
                    curr_dir, "test_data/discrete_action/cartpole_norm.json"
                ),
                "model_output_path": tmpdirname,
                "use_gpu": use_gpu,
                "use_all_avail_gpus": use_all_avail_gpus,
                "init_method": "file://" + lockfile,
                "num_nodes": 1,
                "node_index": 0,
                "actions": ["0", "1"],
                "epochs": 1,
                "rl": {},
                "rainbow": {"double_q_learning": False, "dueling_architecture": False},
                "training": {"minibatch_size": 128},
            }
            dqn_workflow.main(params)
            predictor_files = glob.glob(tmpdirname + "/model_*.torchscript")
            assert len(predictor_files) == 1, "Somehow created two predictor files!"
            predictor = DiscreteDqnTorchPredictor(torch.jit.load(predictor_files[0]))
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
        """Run Parametric DQN workflow to ensure no crashes, algorithm correctness
        not tested here."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            lockfile = os.path.join(tmpdirname, "multiprocess_lock")
            Path(lockfile).touch()
            params = {
                "training_data_path": os.path.join(
                    curr_dir, "test_data/parametric_action/cartpole_training.json.bz2"
                ),
                "eval_data_path": os.path.join(
                    curr_dir, "test_data/parametric_action/cartpole_eval.json.bz2"
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
                "init_method": "file://" + lockfile,
                "num_nodes": 1,
                "node_index": 0,
                "epochs": 1,
                "rl": {},
                "rainbow": {},
                "training": {"minibatch_size": 128},
            }
            parametric_dqn_workflow.main(params)

            predictor_files = glob.glob(tmpdirname + "/model_*.torchscript")
            assert len(predictor_files) == 1, "Somehow created two predictor files!"
            predictor = ParametricDqnTorchPredictor(torch.jit.load(predictor_files[0]))
            test_float_state_features = [{"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0}]
            test_action_features = [{"4": 0.0, "5": 1.0}]
            q_values = predictor.predict(
                test_float_state_features, test_action_features
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

    def test_read_c2_model_from_file(self):
        """Test reading output caffe2 model from file and using it for inference."""
        path = os.path.join(curr_dir, "test_data/discrete_action/example_predictor.c2")
        predictor = DQNPredictor.load(path, "minidb")
        test_float_state_features = [{"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0}]
        q_values = predictor.predict(test_float_state_features)
        assert len(q_values[0].keys()) == 2
