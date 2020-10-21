import random
import unittest

import numpy as np
import pytest
import torch
from reagent.test.ranking.test_seq2slate_utils import (
    MODEL_TRANSFORMER,
    SIMULATION,
    run_seq2slate_tsp,
)


class TestSeq2Slate(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

    def test_seq2slate_transformer_simulation_simple_tsp(self):
        """
        Solve Traveling Salesman Problem. Data comes from one set of nodes (cities).

        Finish in 5 epochs
        """
        device = torch.device("cpu")
        batch_size = 4096
        epochs = 500
        num_batches = 1
        expect_reward_threshold = 1.05
        hidden_size = 32
        num_candidates = 6
        diverse_input = False
        learning_rate = 0.001
        learning_method = SIMULATION
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
    def test_seq2slate_transformer_simulation_hard_tsp(self):
        """
        Solve Traveling Salesman Problem. Data comes from multiple sets of cities.

        4 cities
        batch size=4096, lr=0.001, num batches=300
        """
        device = torch.device("cuda")
        batch_size = 4096
        epochs = 50000
        num_batches = 300
        expect_reward_threshold = 1.04
        hidden_size = 128
        num_candidates = 4
        diverse_input = True
        learning_rate = 0.00005
        learning_method = SIMULATION
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
