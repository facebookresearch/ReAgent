#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
import random
import unittest

import numpy as np
import pytest
import torch
from reagent.test.ranking.test_seq2slate_utils import (
    MODEL_TRANSFORMER,
    OFF_POLICY,
    run_seq2slate_tsp,
)


logger = logging.getLogger(__name__)


class TestSeq2SlateOffPolicy(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

    def test_seq2slate_transformer_off_policy_simple_tsp(self):
        """
        Solve Traveling Salesman Problem. Data comes from one set of nodes (cities).
        """
        device = torch.device("cpu")
        batch_size = 4096
        epochs = 1
        num_batches = 1
        expect_reward_threshold = 1.02
        hidden_size = 32
        num_candidates = 6
        diverse_input = False
        learning_rate = 0.001
        learning_method = OFF_POLICY
        run_seq2slate_tsp(
            MODEL_TRANSFORMER,
            batch_size,
            epochs,
            num_candidates,
            num_batches,
            hidden_size,
            diverse_input,
            learning_rate,
            expect_reward_threshold,
            learning_method,
            device,
        )

    @pytest.mark.seq2slate_long
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_seq2slate_transformer_off_policy_hard_tsp(self):
        """
        Solve Traveling Salesman Problem. Data comes from multiple sets of cities.
        """
        device = torch.device("cuda")
        batch_size = 4096
        epochs = 10
        num_batches = 300
        expect_reward_threshold = 1.02
        hidden_size = 128
        num_candidates = 4
        diverse_input = True
        learning_rate = 0.001
        learning_method = OFF_POLICY
        run_seq2slate_tsp(
            MODEL_TRANSFORMER,
            batch_size,
            epochs,
            num_candidates,
            num_batches,
            hidden_size,
            diverse_input,
            learning_rate,
            expect_reward_threshold,
            learning_method,
            device,
        )
