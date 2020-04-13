#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import json
import logging
import random
import unittest
from typing import List

import numpy as np
import torch
from reagent.json_serialize import json_to_object
from reagent.test.base.horizon_test_base import HorizonTestBase
from reagent.test.gym.run_gym import OpenAiGymParameters
from reagent.test.gym.world_model.state_embed_gym import (
    create_mdnrnn_trainer_and_embed_dataset,
    run_gym,
)


logger = logging.getLogger(__name__)

MDNRNN_STRING_GAME_JSON = "ml/rl/test/configs/mdnrnn_string_game_v0.json"
DQN_STRING_GAME_JSON = "ml/rl/test/configs/discrete_dqn_string_game_v0.json"


class TestStateEmbedGym(HorizonTestBase):
    def setUp(self):
        logging.getLogger().setLevel(logging.INFO)
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        super().setUp()

    @staticmethod
    def verify_result(reward_history: List[float], expected_reward: float):
        assert reward_history[-1] >= expected_reward

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_string_game_gpu(self):
        with open(MDNRNN_STRING_GAME_JSON, "r") as f:
            mdnrnn_params = json_to_object(f.read(), OpenAiGymParameters)
            mdnrnn_params = mdnrnn_params._replace(use_gpu=True)
        with open(DQN_STRING_GAME_JSON, "r") as f:
            rl_params = json_to_object(f.read(), OpenAiGymParameters)
            rl_params = rl_params._replace(use_gpu=True)
        avg_reward_history = self._test_state_embed(mdnrnn_params, rl_params)
        self.verify_result(avg_reward_history, 10)

    @staticmethod
    def _test_state_embed(
        mdnrnn_params: OpenAiGymParameters, rl_params: OpenAiGymParameters
    ):
        env, mdnrnn_trainer, embed_rl_dataset = create_mdnrnn_trainer_and_embed_dataset(
            mdnrnn_params, rl_params.use_gpu
        )
        max_embed_seq_len = mdnrnn_params.run_details.seq_len
        avg_reward_history, _, _, _, _ = run_gym(
            rl_params,
            None,  # score bar
            embed_rl_dataset,
            env.env,
            mdnrnn_trainer.mdnrnn,
            max_embed_seq_len,
        )
        return avg_reward_history
