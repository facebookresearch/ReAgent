#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import json
import logging
import random
import unittest
from typing import Dict, List

import numpy as np
import torch
from reagent.json_serialize import json_to_object
from reagent.test.gym.run_gym import OpenAiGymParameters
from reagent.test.gym.world_model.mdnrnn_gym import mdnrnn_gym


logger = logging.getLogger(__name__)

MDNRNN_CARTPOLE_JSON = "ml/rl/test/configs/mdnrnn_cartpole_v0.json"


class TestMDNRNNGym(unittest.TestCase):
    def setUp(self):
        logging.getLogger().setLevel(logging.DEBUG)
        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)

    @staticmethod
    def verify_result(result_dict: Dict[str, float], expected_top_features: List[str]):
        top_feature = max(result_dict, key=result_dict.get)
        assert (
            top_feature in expected_top_features
        ), f"top_feature: {top_feature}, expected_top_features: {expected_top_features}"

    def test_mdnrnn_cartpole(self):
        with open(MDNRNN_CARTPOLE_JSON, "r") as f:
            params = json_to_object(f.read(), OpenAiGymParameters)
        _, _, feature_importance_map, feature_sensitivity_map, _ = self._test_mdnrnn(
            params, feature_importance=True, feature_sensitivity=True
        )
        self.verify_result(feature_importance_map, ["state1", "state3", "action1"])
        self.verify_result(feature_sensitivity_map, ["state1", "state3"])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_mdnrnn_cartpole_gpu(self):
        with open(MDNRNN_CARTPOLE_JSON, "r") as f:
            params = json_to_object(f.read(), OpenAiGymParameters)
        _, _, feature_importance_map, feature_sensitivity_map, _ = self._test_mdnrnn(
            params, use_gpu=True, feature_importance=True, feature_sensitivity=True
        )
        self.verify_result(feature_importance_map, ["state1", "state3"])
        self.verify_result(feature_sensitivity_map, ["state1", "state3"])

    def _test_mdnrnn(
        self,
        params: OpenAiGymParameters,
        use_gpu=False,
        feature_importance=False,
        feature_sensitivity=False,
    ):
        return mdnrnn_gym(params, feature_importance, feature_sensitivity, seed=0)
