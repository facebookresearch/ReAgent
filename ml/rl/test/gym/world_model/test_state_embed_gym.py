#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import json
import logging
import unittest
from typing import List

import torch
from ml.rl.test.gym.world_model.state_embed_gym import (
    create_mdnrnn_trainer_and_embed_dataset,
    run_gym,
)


logger = logging.getLogger(__name__)

MDNRNN_STRING_GAME_JSON = "ml/rl/test/configs/mdnrnn_string_game_v0.json"
DQN_STRING_GAME_JSON = "ml/rl/test/configs/discrete_dqn_string_game_v0.json"


class TestStateEmbedGym(unittest.TestCase):
    def setUp(self):
        logging.getLogger().setLevel(logging.DEBUG)

    @staticmethod
    def verify_result(reward_history: List[float], expected_reward: float):
        assert reward_history[-1] >= expected_reward

    def test_string_game(self):
        with open(MDNRNN_STRING_GAME_JSON, "r") as f:
            mdnrnn_params = json.load(f)
        with open(DQN_STRING_GAME_JSON, "r") as f:
            rl_params = json.load(f)
        avg_reward_history = self._test_state_embed(
            mdnrnn_params, rl_params, use_gpu=False
        )
        self.verify_result(avg_reward_history, 10)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_string_game_gpu(self):
        with open(MDNRNN_STRING_GAME_JSON, "r") as f:
            mdnrnn_params = json.load(f)
        with open(DQN_STRING_GAME_JSON, "r") as f:
            rl_params = json.load(f)
        avg_reward_history = self._test_state_embed(
            mdnrnn_params, rl_params, use_gpu=True
        )
        self.verify_result(avg_reward_history, 10)

    def _test_state_embed(self, mdnrnn_params, rl_params, use_gpu=False):
        env, mdnrnn_trainer, embed_rl_dataset = create_mdnrnn_trainer_and_embed_dataset(
            mdnrnn_params, use_gpu
        )
        max_embed_seq_len = mdnrnn_params["run_details"]["seq_len"]
        avg_reward_history, _, _, _, _ = run_gym(
            rl_params,
            use_gpu,
            None,  # score bar
            embed_rl_dataset,
            env.env,
            mdnrnn_trainer.mdnrnn,
            max_embed_seq_len,
        )
        return avg_reward_history
