#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import itertools
import logging
import random
import unittest
from collections import defaultdict
from itertools import permutations

import numpy as np
import pytest
import reagent.core.types as rlt
import torch
import torch.nn.functional as F
from parameterized import parameterized
from reagent.model_utils.seq2slate_utils import (
    DECODER_START_SYMBOL,
    Seq2SlateMode,
    Seq2SlateOutputArch,
    mask_logits_by_idx,
    per_symbol_to_per_seq_log_probs,
    per_symbol_to_per_seq_probs,
    subsequent_mask,
    pytorch_decoder_mask,
)
from reagent.test.ranking.seq2slate_utils import (
    MODEL_TRANSFORMER,
    ON_POLICY,
    create_batch,
    create_seq2slate_net,
    rank_on_policy,
    run_seq2slate_tsp,
)


logger = logging.getLogger(__name__)


output_arch_list = [
    Seq2SlateOutputArch.FRECHET_SORT,
    Seq2SlateOutputArch.AUTOREGRESSIVE,
]
temperature_list = [1.0, 2.0]


class TestSeq2SlateOnPolicy(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

    def test_pytorch_decoder_mask(self):
        batch_size = 3
        src_seq_len = 4
        num_heads = 2

        memory = torch.randn(batch_size, src_seq_len, num_heads)
        tgt_in_idx = torch.tensor([[1, 2, 3], [1, 4, 2], [1, 5, 4]]).long()
        tgt_tgt_mask, tgt_src_mask = pytorch_decoder_mask(memory, tgt_in_idx, num_heads)

        expected_tgt_tgt_mask = (
            torch.tensor(
                [
                    [False, True, True],
                    [False, False, True],
                    [False, False, False],
                ],
            )
            .unsqueeze(0)
            .repeat(batch_size * num_heads, 1, 1)
        )
        expected_tgt_src_mask = torch.tensor(
            [
                [
                    [False, False, False, False],
                    [True, False, False, False],
                    [True, True, False, False],
                ],
                [
                    [False, False, False, False],
                    [False, False, True, False],
                    [True, False, True, False],
                ],
                [
                    [False, False, False, False],
                    [False, False, False, True],
                    [False, False, True, True],
                ],
            ]
        ).repeat_interleave(num_heads, dim=0)
        assert torch.all(tgt_tgt_mask == expected_tgt_tgt_mask)
        assert torch.all(tgt_src_mask == expected_tgt_src_mask)

    def test_per_symbol_to_per_seq_log_probs(self):
        """
        Test per_symbol_to_per_seq_log_probs method
        """
        batch_size = 1
        seq_len = 3
        candidate_size = seq_len + 2

        tgt_out_idx = torch.tensor([[0, 2, 1]]) + 2
        per_symbol_log_probs = torch.randn(batch_size, seq_len, candidate_size)
        per_symbol_log_probs[0, :, :2] = float("-inf")
        per_symbol_log_probs[0, 1, 2] = float("-inf")
        per_symbol_log_probs[0, 2, 2] = float("-inf")
        per_symbol_log_probs[0, 2, 4] = float("-inf")
        per_symbol_log_probs = F.log_softmax(per_symbol_log_probs, dim=2)

        expect_per_seq_log_probs = (
            per_symbol_log_probs[0, 0, 2]
            + per_symbol_log_probs[0, 1, 4]
            + per_symbol_log_probs[0, 2, 3]
        )
        computed_per_seq_log_probs = per_symbol_to_per_seq_log_probs(
            per_symbol_log_probs, tgt_out_idx
        )
        np.testing.assert_allclose(
            expect_per_seq_log_probs, computed_per_seq_log_probs, atol=0.001, rtol=0.0
        )

    def test_per_symbol_to_per_seq_probs(self):
        batch_size = 1
        seq_len = 3
        candidate_size = seq_len + 2

        tgt_out_idx = torch.tensor([[0, 2, 1]]) + 2
        per_symbol_log_probs = torch.randn(batch_size, seq_len, candidate_size)
        per_symbol_log_probs[0, :, :2] = float("-inf")
        per_symbol_log_probs[0, 1, 2] = float("-inf")
        per_symbol_log_probs[0, 2, 2] = float("-inf")
        per_symbol_log_probs[0, 2, 4] = float("-inf")
        per_symbol_log_probs = F.log_softmax(per_symbol_log_probs, dim=2)
        per_symbol_probs = torch.exp(per_symbol_log_probs)

        expect_per_seq_probs = (
            per_symbol_probs[0, 0, 2]
            * per_symbol_probs[0, 1, 4]
            * per_symbol_probs[0, 2, 3]
        )
        computed_per_seq_probs = per_symbol_to_per_seq_probs(
            per_symbol_probs, tgt_out_idx
        )
        np.testing.assert_allclose(
            expect_per_seq_probs, computed_per_seq_probs, atol=0.001, rtol=0.0
        )

    def test_subsequent_mask(self):
        expect_mask = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        mask = subsequent_mask(3, torch.device("cpu"))
        assert torch.all(torch.eq(mask, expect_mask))

    def test_mask_logits_by_idx(self):
        logits = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [2.0, 3.0, 4.0, 5.0, 6.0],
                    [3.0, 4.0, 5.0, 6.0, 7.0],
                ],
                [
                    [5.0, 4.0, 3.0, 2.0, 1.0],
                    [6.0, 5.0, 4.0, 3.0, 2.0],
                    [7.0, 6.0, 5.0, 4.0, 3.0],
                ],
            ]
        )
        tgt_in_idx = torch.tensor(
            [[DECODER_START_SYMBOL, 2, 3], [DECODER_START_SYMBOL, 4, 3]]
        )
        masked_logits = mask_logits_by_idx(logits, tgt_in_idx)
        expected_logits = torch.tensor(
            [
                [
                    [float("-inf"), float("-inf"), 3.0, 4.0, 5.0],
                    [float("-inf"), float("-inf"), float("-inf"), 5.0, 6.0],
                    [float("-inf"), float("-inf"), float("-inf"), float("-inf"), 7.0],
                ],
                [
                    [float("-inf"), float("-inf"), 3.0, 2.0, 1.0],
                    [float("-inf"), float("-inf"), 4.0, 3.0, float("-inf")],
                    [float("-inf"), float("-inf"), 5.0, float("-inf"), float("-inf")],
                ],
            ]
        )
        assert torch.all(torch.eq(masked_logits, expected_logits))

    @parameterized.expand(itertools.product(output_arch_list, temperature_list))
    @torch.no_grad()
    def test_seq2slate_transformer_propensity_computation(
        self, output_arch, temperature
    ):
        """
        Test propensity computation of seq2slate net
        """
        candidate_num = 4
        candidate_dim = 2
        hidden_size = 32
        all_perm = torch.tensor(
            list(permutations(torch.arange(candidate_num), candidate_num))
        )
        batch_size = len(all_perm)
        device = torch.device("cpu")

        seq2slate_net = create_seq2slate_net(
            MODEL_TRANSFORMER,
            candidate_num,
            candidate_dim,
            hidden_size,
            output_arch,
            temperature,
            device,
        )
        batch = create_batch(
            batch_size,
            candidate_num,
            candidate_dim,
            device,
            ON_POLICY,
            diverse_input=False,
        )
        batch = rlt.PreprocessedRankingInput.from_input(
            state=batch.state.float_features,
            candidates=batch.src_seq.float_features,
            device=device,
            action=all_perm,
        )
        per_symbol_log_prob = seq2slate_net(
            batch, mode=Seq2SlateMode.PER_SYMBOL_LOG_PROB_DIST_MODE
        ).log_probs
        per_seq_log_prob = seq2slate_net(
            batch, mode=Seq2SlateMode.PER_SEQ_LOG_PROB_MODE
        ).log_probs
        per_seq_log_prob_computed = per_symbol_to_per_seq_log_probs(
            per_symbol_log_prob, all_perm + 2
        )
        # probabilities of two modes should match
        np.testing.assert_allclose(
            per_seq_log_prob, per_seq_log_prob_computed, atol=0.00001
        )
        # probabilities of all possible permutations should sum up to 1
        np.testing.assert_allclose(
            torch.sum(torch.exp(per_seq_log_prob)), 1.0, atol=0.00001
        )

    @parameterized.expand(itertools.product(output_arch_list, temperature_list))
    def test_seq2slate_transformer_onpolicy_basic_logic(self, output_arch, temperature):
        """
        Test basic logic of seq2slate on policy sampling
        """
        device = torch.device("cpu")
        candidate_num = 4
        candidate_dim = 2
        batch_size = 4096
        hidden_size = 32
        seq2slate_net = create_seq2slate_net(
            MODEL_TRANSFORMER,
            candidate_num,
            candidate_dim,
            hidden_size,
            output_arch,
            temperature,
            device,
        )
        batch = create_batch(
            batch_size,
            candidate_num,
            candidate_dim,
            device,
            ON_POLICY,
            diverse_input=False,
        )

        action_to_propensity_map = {}
        action_count = defaultdict(int)
        total_count = 0
        for i in range(50):
            model_propensity, model_action = rank_on_policy(
                seq2slate_net, batch, candidate_num, greedy=False
            )
            for propensity, action in zip(model_propensity, model_action):
                action_str = ",".join(map(str, action.numpy().tolist()))

                # Same action always leads to same propensity
                if action_to_propensity_map.get(action_str) is None:
                    action_to_propensity_map[action_str] = float(propensity)
                else:
                    np.testing.assert_allclose(
                        action_to_propensity_map[action_str],
                        float(propensity),
                        atol=0.001,
                        rtol=0.0,
                    )

                action_count[action_str] += 1
                total_count += 1

            logger.info(f"Finish {i} round, {total_count} data counts")

        # Check action distribution
        for action_str, count in action_count.items():
            empirical_propensity = count / total_count
            computed_propensity = action_to_propensity_map[action_str]
            logger.info(
                f"action={action_str}, empirical propensity={empirical_propensity}, "
                f"computed propensity={computed_propensity}"
            )
            np.testing.assert_allclose(
                computed_propensity, empirical_propensity, atol=0.01, rtol=0.0
            )

    def test_seq2slate_transformer_on_policy_simple_tsp(self):
        """
        Solve Traveling Salesman Problem. Cities comes from a fixed set of nodes (cities).
        Easily hit reward threshold after one batch training
        """
        device = torch.device("cpu")
        batch_size = 4096
        epochs = 1
        num_batches = 50
        expect_reward_threshold = 1.02
        hidden_size = 32
        num_candidates = 6
        diverse_input = False
        learning_rate = 0.001
        learning_method = ON_POLICY
        policy_gradient_interval = 1
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
            policy_gradient_interval,
            device,
        )

    @pytest.mark.seq2slate_long
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_seq2slate_transformer_on_policy_hard_tsp(self):
        """
        Solve Traveling Salesman Problem. Data comes from different sets of cities.
        """
        device = torch.device("cuda")
        batch_size = 4096
        epochs = 3
        num_batches = 300
        expect_reward_threshold = 1.03
        hidden_size = 32
        num_candidates = 6
        diverse_input = True
        learning_rate = 0.001
        learning_method = ON_POLICY
        policy_gradient_interval = 1
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
            policy_gradient_interval,
            device,
        )
