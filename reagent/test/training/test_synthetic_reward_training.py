#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import torch
from reagent.core import types as rlt
from reagent.models.synthetic_reward import SingleStepSyntheticRewardNet
from reagent.optimizer.union import Optimizer__Union
from reagent.optimizer.union import classes
from reagent.training import RewardNetTrainer


def create_data(state_dim, action_dim, seq_len, batch_size, num_batches):
    SCALE = 2
    weight = SCALE * torch.randn(state_dim + action_dim)

    def data_generator():
        for _ in range(num_batches):
            state = SCALE * torch.randn(seq_len, batch_size, state_dim)
            action = SCALE * torch.randn(seq_len, batch_size, action_dim)
            # random valid step
            valid_step = torch.randint(1, seq_len + 1, (batch_size, 1))

            # reward_matrix shape: batch_size x seq_len
            reward_matrix = torch.matmul(
                torch.cat((state, action), dim=2), weight
            ).transpose(0, 1)
            mask = torch.arange(seq_len).repeat(batch_size, 1)
            mask = (mask >= (seq_len - valid_step)).float()
            reward = (reward_matrix * mask).sum(dim=1).reshape(-1, 1)
            input = rlt.MemoryNetworkInput(
                state=rlt.FeatureData(state),
                action=action,
                valid_step=valid_step,
                reward=reward,
                # the rest fields will not be used
                next_state=torch.tensor([]),
                step=torch.tensor([]),
                not_terminal=torch.tensor([]),
                time_diff=torch.tensor([]),
            )
            yield input

    return weight, data_generator


class TestSyntheticRewardTraining(unittest.TestCase):
    def test_linear_reward_parametric_reward(self):
        """
        Reward at each step is a linear function of state and action.
        However, we can only observe aggregated reward at the last step
        """
        state_dim = 10
        action_dim = 2
        seq_len = 5
        batch_size = 512
        num_batches = 10000
        sizes = [256, 128]
        activations = ["relu", "relu"]
        last_layer_activation = "linear"
        reward_net = SingleStepSyntheticRewardNet(
            state_dim=state_dim,
            action_dim=action_dim,
            sizes=sizes,
            activations=activations,
            last_layer_activation=last_layer_activation,
        )
        optimizer = Optimizer__Union(SGD=classes["SGD"]())
        trainer = RewardNetTrainer(reward_net, optimizer)

        weight, data_generator = create_data(
            state_dim, action_dim, seq_len, batch_size, num_batches
        )
        threshold = 0.1
        reach_threshold = False
        for batch in data_generator():
            loss = trainer.train(batch)
            if loss < threshold:
                reach_threshold = True
                break

        assert reach_threshold
