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
        epochs = 500
        num_batches = 30
        expect_reward_threshold = 1.05
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

        Tried several experiment settings and the current one takes least time to finish:
        (current) random logging, scale reward, reaches 9.72 in 5400 batches
        random logging, not scale reward, reaches 10.09 after 5400 batches
        frechet sort shape 0.1 logging, scale reward, reaches 9.59 in 3300 batches
        frechet sort shape 0.5 logging, scale reward, reaches 9.6 in 7500 batches
        """
        device = torch.device("cuda")
        batch_size = 4096
        epochs = 50000
        num_batches = 300
        expect_reward_threshold = 1.06
        hidden_size = 128
        num_candidates = 4
        diverse_input = True
        learning_rate = 0.00005
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
