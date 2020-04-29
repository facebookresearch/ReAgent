#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
Environments that require short training and evaluation time (<=10min)
can be tested in this file.
"""
import logging
import os
import random
import unittest

import numpy as np
import torch
from reagent.json_serialize import json_to_object
from reagent.tensorboardX import SummaryWriterContext
from reagent.test.gym.run_gym import OpenAiGymParameters, run_gym


curr_dir = os.path.dirname(__file__)


DQN_CARTPOLE_JSON = os.path.join(curr_dir, "discrete_dqn_cartpole_v0.json")
# Though maximal score is 200, we set a lower bar to let tests finish in time
CARTPOLE_SCORE_BAR = 100
SEED = 0


class TestGym(unittest.TestCase):
    def setUp(self):
        logging.getLogger().setLevel(logging.INFO)
        SummaryWriterContext._reset_globals()
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        random.seed(SEED)

    @unittest.skip("Replaced with new gym tests.. Remove when possible.")
    def test_dqn_cartpole_offline(self):
        """ Test if the json config works for offline DQN in Cartpole """
        with open(DQN_CARTPOLE_JSON, "r") as f:
            params = json_to_object(f.read(), OpenAiGymParameters)
        reward_history, _, _, _, _ = run_gym(
            params, offline_train=True, score_bar=CARTPOLE_SCORE_BAR, seed=SEED
        )
        assert reward_history[-1] >= CARTPOLE_SCORE_BAR

    @unittest.skip("Replaced with new gym tests.. Remove when possible.")
    def test_dqn_cartpole_online(self):
        """ Test if the json config works for online DQN in Cartpole """
        with open(DQN_CARTPOLE_JSON, "r") as f:
            params = json_to_object(f.read(), OpenAiGymParameters)
        reward_history, _, _, _, _ = run_gym(
            params, offline_train=False, score_bar=CARTPOLE_SCORE_BAR, seed=SEED
        )
        assert reward_history[-1] > CARTPOLE_SCORE_BAR
