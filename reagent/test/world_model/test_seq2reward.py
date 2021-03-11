#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest
from typing import Optional

import torch
import torch.nn as nn
from reagent.core import types as rlt
from reagent.prediction.predictor_wrapper import (
    Seq2RewardWithPreprocessor,
    Seq2RewardPlanShortSeqWithPreprocessor,
    FAKE_STATE_ID_LIST_FEATURES,
    FAKE_STATE_ID_SCORE_LIST_FEATURES,
)
from reagent.preprocessing.identify_types import DO_NOT_PREPROCESS
from reagent.preprocessing.normalization import NormalizationParameters
from reagent.preprocessing.preprocessor import Preprocessor
from reagent.training.utils import gen_permutations
from reagent.training.world_model.seq2reward_trainer import get_Q

logger = logging.getLogger(__name__)


class FakeStepPredictionNetwork(nn.Module):
    def __init__(self, look_ahead_steps):
        super().__init__()
        self.look_ahead_steps = look_ahead_steps

    def forward(self, state: torch.Tensor):
        """
        Given the current state, predict the probability of
        experiencing next n steps (1 <=n <= look_ahead_steps)

        For the test purpose, it outputs fixed fake numbers
        """
        batch_size, _ = state.shape
        return torch.ones(batch_size, self.look_ahead_steps).float()


class FakeSeq2RewardNetwork(nn.Module):
    def forward(
        self,
        state: rlt.FeatureData,
        action: rlt.FeatureData,
        valid_reward_len: Optional[torch.Tensor] = None,
    ):
        """
        Mimic I/O of Seq2RewardNetwork but return fake reward
        Reward is the concatenation of action indices, independent
        of state.

        For example, when seq_len = 3, batch_size = 1, action_num = 2,
        acc_reward = tensor(
            [[  0.],
            [  1.],
            [ 10.],
            [ 11.],
            [100.],
            [101.],
            [110.],
            [111.]]
        )

        Input action shape: seq_len, batch_size, num_action
        Output acc_reward shape: batch_size, 1
        """
        # pyre-fixme[9]: action has type `FeatureData`; used as `Tensor`.
        action = action.float_features.transpose(0, 1)
        action_indices = torch.argmax(action, dim=2).tolist()
        acc_reward = torch.tensor(
            list(map(lambda x: float("".join(map(str, x))), action_indices))
        ).reshape(-1, 1)
        logger.info(f"acc_reward: {acc_reward}")
        return rlt.Seq2RewardOutput(acc_reward=acc_reward)


class TestSeq2Reward(unittest.TestCase):
    def test_seq2reward_with_preprocessor_plan_short_sequence(self):
        self._test_seq2reward_with_preprocessor(plan_short_sequence=True)

    def test_seq2reward_with_preprocessor_plan_full_sequence(self):
        self._test_seq2reward_with_preprocessor(plan_short_sequence=False)

    def _test_seq2reward_with_preprocessor(self, plan_short_sequence):
        state_dim = 4
        action_dim = 2
        seq_len = 3
        model = FakeSeq2RewardNetwork()
        state_normalization_parameters = {
            i: NormalizationParameters(
                feature_type=DO_NOT_PREPROCESS, mean=0.0, stddev=1.0
            )
            for i in range(1, state_dim)
        }
        state_preprocessor = Preprocessor(state_normalization_parameters, False)

        if plan_short_sequence:
            step_prediction_model = FakeStepPredictionNetwork(seq_len)
            model_with_preprocessor = Seq2RewardPlanShortSeqWithPreprocessor(
                model,
                step_prediction_model,
                state_preprocessor,
                seq_len,
                action_dim,
            )
        else:
            model_with_preprocessor = Seq2RewardWithPreprocessor(
                model,
                state_preprocessor,
                seq_len,
                action_dim,
            )
        input_prototype = rlt.ServingFeatureData(
            float_features_with_presence=state_preprocessor.input_prototype(),
            id_list_features=FAKE_STATE_ID_LIST_FEATURES,
            id_score_list_features=FAKE_STATE_ID_SCORE_LIST_FEATURES,
        )
        q_values = model_with_preprocessor(input_prototype)
        if plan_short_sequence:
            # When planning for 1, 2, and 3 steps ahead,
            # the expected q values are respectively:
            # [0, 1], [1, 11], [11, 111]
            # Weighting the expected q values by predicted step
            # probabilities [0.33, 0.33, 0.33], we have [4, 41]
            expected_q_values = torch.tensor([[4.0, 41.0]])
        else:
            expected_q_values = torch.tensor([[11.0, 111.0]])
        assert torch.all(expected_q_values == q_values)

    def test_get_Q(self):
        NUM_ACTION = 2
        MULTI_STEPS = 3
        BATCH_SIZE = 2
        STATE_DIM = 4
        all_permut = gen_permutations(MULTI_STEPS, NUM_ACTION)
        seq2reward_network = FakeSeq2RewardNetwork()
        state = torch.zeros(BATCH_SIZE, STATE_DIM)
        q_values = get_Q(seq2reward_network, state, all_permut)
        expected_q_values = torch.tensor([[11.0, 111.0], [11.0, 111.0]])
        logger.info(f"q_values: {q_values}")
        assert torch.all(expected_q_values == q_values)

    def test_gen_permutations_seq_len_1_action_6(self):
        SEQ_LEN = 1
        NUM_ACTION = 6
        expected_outcome = torch.tensor([[0], [1], [2], [3], [4], [5]])
        self._test_gen_permutations(SEQ_LEN, NUM_ACTION, expected_outcome)

    def test_gen_permutations_seq_len_3_num_action_2(self):
        SEQ_LEN = 3
        NUM_ACTION = 2
        expected_outcome = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        self._test_gen_permutations(SEQ_LEN, NUM_ACTION, expected_outcome)

    def _test_gen_permutations(self, SEQ_LEN, NUM_ACTION, expected_outcome):
        # expected shape: SEQ_LEN, PERM_NUM, ACTION_DIM
        result = gen_permutations(SEQ_LEN, NUM_ACTION)
        assert result.shape == (SEQ_LEN, NUM_ACTION ** SEQ_LEN, NUM_ACTION)
        outcome = torch.argmax(result.transpose(0, 1), dim=-1)
        assert torch.all(outcome == expected_outcome)
