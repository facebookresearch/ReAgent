#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest
from typing import Optional

import torch
import torch.nn as nn
from reagent import types as rlt
from reagent.training.utils import gen_permutations
from reagent.training.world_model.seq2reward_trainer import get_Q


logger = logging.getLogger(__name__)


class FakeSeq2RewardNetwork(nn.Module):
    def __init__(self):
        super().__init__()

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
    def test_get_Q(self):
        NUM_ACTION = 2
        MULTI_STEPS = 3
        BATCH_SIZE = 2
        STATE_DIM = 4
        all_permut = gen_permutations(MULTI_STEPS, NUM_ACTION)
        seq2reward_network = FakeSeq2RewardNetwork()
        batch = rlt.MemoryNetworkInput(
            state=rlt.FeatureData(
                float_features=torch.zeros(MULTI_STEPS, BATCH_SIZE, STATE_DIM)
            ),
            next_state=rlt.FeatureData(
                float_features=torch.zeros(MULTI_STEPS, BATCH_SIZE, STATE_DIM)
            ),
            action=rlt.FeatureData(
                float_features=torch.zeros(MULTI_STEPS, BATCH_SIZE, NUM_ACTION)
            ),
            reward=torch.zeros(1),
            time_diff=torch.zeros(1),
            step=torch.zeros(1),
            not_terminal=torch.zeros(1),
        )
        q_values = get_Q(seq2reward_network, batch, all_permut)
        expected_q_values = torch.tensor([[11.0, 111.0], [11.0, 111.0]])
        logger.info(f"q_values: {q_values}")
        assert torch.all(expected_q_values == q_values)

    def test_gen_permutations(self):
        SEQ_LEN = 3
        NUM_ACTION = 2
        # expected shape: SEQ_LEN, PERM_NUM, ACTION_DIM
        result = gen_permutations(SEQ_LEN, NUM_ACTION)
        assert result.shape == (SEQ_LEN, NUM_ACTION ** SEQ_LEN, NUM_ACTION)
        outcome = torch.argmax(result.transpose(0, 1), dim=-1)
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
        assert torch.all(outcome == expected_outcome)
