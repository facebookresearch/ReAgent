#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import random
import unittest
from collections import defaultdict
from itertools import permutations

import numpy as np
import pytest
import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.model_utils.seq2slate_utils import (
    Seq2SlateMode,
    per_symbol_to_per_seq_log_probs,
)
from reagent.models.seq2slate import Seq2SlateMode, Seq2SlateTransformerNet
from reagent.optimizer.union import Optimizer__Union
from reagent.parameters import Seq2SlateParameters
from reagent.torch_utils import gather
from reagent.training.ranking.seq2slate_trainer import Seq2SlateTrainer


logger = logging.getLogger(__name__)

MODEL_TRANSFORMER = "transformer"


def create_batch(batch_size, candidate_num, candidate_dim, device, diverse_input=False):
    state = torch.zeros(batch_size, 1)  # fake state, we only use candidates
    # # city coordinates are spread in [0, 4]
    candidates = torch.randint(5, (batch_size, candidate_num, candidate_dim)).float()
    if not diverse_input:
        # every training data has the same nodes as the input cities
        candidates[1:] = candidates[0]
    batch = rlt.PreprocessedRankingInput.from_input(
        state=state.to(device), candidates=candidates.to(device), device=device
    )
    return batch


def compute_reward(ranked_cities):
    assert len(ranked_cities.shape) == 3
    ranked_cities_offset = torch.roll(ranked_cities, shifts=1, dims=1)
    return (
        torch.sqrt(((ranked_cities_offset - ranked_cities) ** 2).sum(-1))
        .sum(-1)
        .unsqueeze(1)
    )


def compute_best_reward(input_cities):
    batch_size, candidate_num, _ = input_cities.shape
    all_perm = torch.tensor(
        list(permutations(torch.arange(candidate_num), candidate_num))
    )
    res = [
        compute_reward(gather(input_cities, perm.repeat(batch_size, 1)))
        for perm in all_perm
    ]
    # res shape: batch_size, num_perm
    res = torch.cat(res, dim=1)
    best_possible_reward = torch.min(res, dim=1).values
    best_possible_reward_mean = torch.mean(best_possible_reward)
    return best_possible_reward_mean


@torch.no_grad()
def rank_on_policy(
    model, batch: rlt.PreprocessedRankingInput, tgt_seq_len: int, greedy: bool
):
    model.eval()
    rank_output = model(
        batch, mode=Seq2SlateMode.RANK_MODE, tgt_seq_len=tgt_seq_len, greedy=greedy
    )
    ranked_slate_prob = torch.prod(
        torch.gather(
            rank_output.ranked_tgt_out_probs,
            2,
            rank_output.ranked_tgt_out_idx.unsqueeze(-1),
        ).squeeze(),
        dim=-1,
        keepdim=True,
    )
    ranked_order = rank_output.ranked_tgt_out_idx - 2
    model.train()
    return ranked_slate_prob, ranked_order


@torch.no_grad()
def rank_on_policy_and_eval(
    seq2slate_net, batch: rlt.PreprocessedRankingInput, tgt_seq_len: int, greedy: bool
):
    model_propensity, model_action = rank_on_policy(
        seq2slate_net, batch, tgt_seq_len, greedy=greedy
    )
    ranked_cities = gather(batch.src_seq.float_features, model_action)
    reward = compute_reward(ranked_cities)
    return model_propensity, model_action, reward


def create_seq2slate_transformer(candidate_num, candidate_dim, hidden_size, device):
    return Seq2SlateTransformerNet(
        state_dim=1,
        candidate_dim=candidate_dim,
        num_stacked_layers=2,
        num_heads=2,
        dim_model=hidden_size,
        dim_feedforward=hidden_size,
        max_src_seq_len=candidate_num,
        max_tgt_seq_len=candidate_num,
        encoder_only=False,
    ).to(device)


def create_trainer(seq2slate_net, batch_size, learning_rate, device, on_policy):
    use_gpu = False if device == torch.device("cpu") else True
    return Seq2SlateTrainer(
        seq2slate_net=seq2slate_net,
        minibatch_size=batch_size,
        parameters=Seq2SlateParameters(on_policy=on_policy),
        policy_optimizer=Optimizer__Union.default(lr=learning_rate),
        use_gpu=use_gpu,
        print_interval=100,
    )


class TestSeq2Slate(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

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

    @torch.no_grad()
    def test_seq2slate_transformer_propensity_computation(self):
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

        seq2slate_net = create_seq2slate_transformer(
            candidate_num, candidate_dim, hidden_size, device
        )
        batch = create_batch(
            batch_size, candidate_num, candidate_dim, device, diverse_input=False
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

    def test_seq2slate_transformer_onplicy_basic_logic(self):
        """
        Test basic logic of seq2slate on policy sampling
        """
        device = torch.device("cpu")
        candidate_num = 4
        candidate_dim = 2
        batch_size = 4096
        hidden_size = 32
        seq2slate_net = create_seq2slate_transformer(
            candidate_num, candidate_dim, hidden_size, device
        )
        batch = create_batch(
            batch_size, candidate_num, candidate_dim, device, diverse_input=False
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
        self._test_seq2slate_on_policy_tsp(
            MODEL_TRANSFORMER,
            batch_size,
            epochs,
            num_candidates,
            num_batches,
            hidden_size,
            diverse_input,
            learning_rate,
            expect_reward_threshold,
            device,
        )

    @pytest.mark.seq2slate_long
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_seq2slate_transformer_on_policy_hard_tsp(self):
        """
        Solve Traveling Salesman Problem. Data comes from multiple sets of cities.

        4 cities
        batch size 512, lr=0.00005, num batches 300: 1788sec
        batch size 4096, lr=0.00005, num batch 300: 917 sec
        batch size 4096, lr=0.00005, num batch 150: 948 sec
        batch size 8192, lr=0.0001, num batch 100: 1166 sec
        batch size 8192, lr=0.00005, num batch 100: 817 sec
        batch size 10240, lr=0.00005, num batch 100: 1828 sec
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
        self._test_seq2slate_on_policy_tsp(
            MODEL_TRANSFORMER,
            batch_size,
            epochs,
            num_candidates,
            num_batches,
            hidden_size,
            diverse_input,
            learning_rate,
            expect_reward_threshold,
            device,
        )

    def _test_seq2slate_on_policy_tsp(
        self,
        model_str,
        batch_size,
        epochs,
        candidate_num,
        num_batches,
        hidden_size,
        diverse_input,
        learning_rate,
        expect_reward_threshold,
        device,
    ):
        candidate_dim = 2
        eval_sample_size = 1

        batch_list = [
            create_batch(
                batch_size,
                candidate_num,
                candidate_dim,
                device,
                diverse_input=diverse_input,
            )
            for _ in range(num_batches)
        ]

        if diverse_input:
            test_batch = create_batch(
                batch_size,
                candidate_num,
                candidate_dim,
                device,
                diverse_input=diverse_input,
            )
        else:
            test_batch = batch_list[0]

        best_test_possible_reward = compute_best_reward(
            test_batch.src_seq.float_features
        )

        if model_str == MODEL_TRANSFORMER:
            seq2slate_net = create_seq2slate_transformer(
                candidate_num, candidate_dim, hidden_size, device
            )
        else:
            raise NotImplementedError(f"unknown model type {model_str}")

        trainer = create_trainer(
            seq2slate_net, batch_size, learning_rate, device, on_policy=True
        )

        for e in range(epochs):
            for batch in batch_list:
                model_propensity, model_action, reward = rank_on_policy_and_eval(
                    seq2slate_net, batch, candidate_num, greedy=False
                )
                on_policy_batch = rlt.PreprocessedRankingInput.from_input(
                    state=batch.state.float_features,
                    candidates=batch.src_seq.float_features,
                    device=device,
                    action=model_action,
                    logged_propensities=model_propensity,
                    slate_reward=-reward,  # negate because we want to minimize
                )
                trainer.train(
                    rlt.PreprocessedTrainingBatch(training_input=on_policy_batch)
                )
                logger.info(f"Epoch {e} mean on_policy reward: {torch.mean(reward)}")
                logger.info(
                    f"Epoch {e} mean model_propensity: {torch.mean(model_propensity)}"
                )

            # evaluation
            best_test_reward = torch.full((batch_size,), 1e9).to(device)
            for _ in range(eval_sample_size):
                _, _, reward = rank_on_policy_and_eval(
                    seq2slate_net, test_batch, candidate_num, greedy=True
                )
                best_test_reward = torch.where(
                    reward < best_test_reward, reward, best_test_reward
                )
            logger.info(
                f"Test mean reward: {torch.mean(best_test_reward)}, "
                f"best possible reward {best_test_possible_reward}"
            )
            if (
                torch.mean(best_test_reward)
                < best_test_possible_reward * expect_reward_threshold
            ):
                return

        raise AssertionError(
            "Test failed because it did not reach expected test reward"
        )
