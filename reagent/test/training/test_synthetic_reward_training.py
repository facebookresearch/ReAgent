#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

import pytorch_lightning as pl
import torch
from reagent.core import parameters as rlp, types as rlt
from reagent.models.synthetic_reward import (
    NGramConvolutionalNetwork,
    NGramFullyConnectedNetwork,
    SequenceSyntheticRewardNet,
    SingleStepSyntheticRewardNet,
    SyntheticRewardNet,
    TransformerSyntheticRewardNet,
)
from reagent.optimizer.union import classes, Optimizer__Union
from reagent.reporting.reward_network_reporter import RewardNetworkReporter
from reagent.training import RewardNetTrainer
from reagent.training.reward_network_trainer import LossFunction
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


def create_data(
    state_dim, action_dim, seq_len, batch_size, num_batches, binary_reward=False
):
    SCALE = 2
    # reward is a linear function of (state, action)
    weight = SCALE * torch.randn(state_dim + action_dim)
    data = [None for _ in range(num_batches)]
    for i in range(num_batches):
        state = SCALE * torch.randn(seq_len, batch_size, state_dim)
        action = SCALE * torch.randn(seq_len, batch_size, action_dim)
        # random valid step
        valid_step = torch.randint(1, seq_len + 1, (batch_size, 1))

        # reward_matrix shape: batch_size x seq_len
        reward_matrix = torch.matmul(
            torch.cat((state, action), dim=2), weight
        ).transpose(0, 1)
        if binary_reward:
            reward_matrix = torch.sigmoid(reward_matrix)
        mask = torch.arange(seq_len).repeat(batch_size, 1)
        mask = (mask >= (seq_len - valid_step)).float()
        reward = (reward_matrix * mask).sum(dim=1).reshape(-1, 1)
        data[i] = rlt.MemoryNetworkInput(
            state=rlt.FeatureData(state),
            action=rlt.FeatureData(action),
            valid_step=valid_step,
            reward=reward,
            # the rest fields will not be used
            next_state=torch.tensor([]),
            step=torch.tensor([]),
            not_terminal=torch.tensor([]),
            time_diff=torch.tensor([]),
        )
    return weight, data


def create_sequence_data(state_dim, action_dim, seq_len, batch_size, num_batches):
    SCALE = 2
    weight = SCALE * torch.randn(state_dim + action_dim)

    data = [None for _ in range(num_batches)]

    for i in range(num_batches):
        state = SCALE * torch.randn(seq_len, batch_size, state_dim)
        action = SCALE * torch.randn(seq_len, batch_size, action_dim)
        # random valid step
        valid_step = torch.randint(1, seq_len + 1, (batch_size, 1))

        feature_mask = torch.arange(seq_len).repeat(batch_size, 1)
        feature_mask = (feature_mask >= (seq_len - valid_step)).float()
        assert feature_mask.shape == (batch_size, seq_len), feature_mask.shape
        feature_mask = feature_mask.transpose(0, 1).unsqueeze(-1)
        assert feature_mask.shape == (seq_len, batch_size, 1), feature_mask.shape

        feature = torch.cat((state, action), dim=2)
        masked_feature = feature * feature_mask

        # seq_len, batch_size, state_dim + action_dim
        left_shifted = torch.cat(
            (
                masked_feature.narrow(0, 1, seq_len - 1),
                torch.zeros(1, batch_size, state_dim + action_dim),
            ),
            dim=0,
        )
        # seq_len, batch_size, state_dim + action_dim
        right_shifted = torch.cat(
            (
                torch.zeros(1, batch_size, state_dim + action_dim),
                masked_feature.narrow(0, 0, seq_len - 1),
            ),
            dim=0,
        )
        # reward_matrix shape: batch_size x seq_len
        reward_matrix = torch.matmul(left_shifted + right_shifted, weight).transpose(
            0, 1
        )

        mask = torch.arange(seq_len).repeat(batch_size, 1)
        mask = (mask >= (seq_len - valid_step)).float()
        reward = (reward_matrix * mask).sum(dim=1).reshape(-1, 1)

        data[i] = rlt.MemoryNetworkInput(
            state=rlt.FeatureData(state),
            action=rlt.FeatureData(action),
            valid_step=valid_step,
            reward=reward,
            # the rest fields will not be used
            next_state=torch.tensor([]),
            step=torch.tensor([]),
            not_terminal=torch.tensor([]),
            time_diff=torch.tensor([]),
        )

    return weight, data


def train_and_eval(trainer, data, num_eval_batches=100, max_epochs=1):
    train_dataloader = DataLoader(data[:-num_eval_batches], collate_fn=lambda x: x[0])
    eval_data = data[-num_eval_batches:]

    # disable logging in tests
    pl_trainer = pl.Trainer(max_epochs=max_epochs, logger=False)
    pl_trainer.fit(trainer, train_dataloader)

    total_loss = 0
    for i, batch in enumerate(eval_data):
        loss = trainer.validation_step(batch, batch_idx=i)
        total_loss += loss
    return total_loss / num_eval_batches


class TestSyntheticRewardTraining(unittest.TestCase):
    def setUp(self):
        pl.seed_everything(123)

    def test_linear_reward_parametric_reward_success(self):
        avg_eval_loss = self._test_linear_reward_parametric_reward(
            ground_truth_reward_from_multiple_steps=False
        )
        threshold = 0.1
        assert avg_eval_loss < threshold

    def test_linear_reward_parametric_reward_fail(self):
        avg_eval_loss = self._test_linear_reward_parametric_reward(
            ground_truth_reward_from_multiple_steps=True
        )
        # fail to learn
        threshold = 100.0
        assert avg_eval_loss > threshold

    def _test_linear_reward_parametric_reward(
        self, ground_truth_reward_from_multiple_steps=False
    ):
        """
        Reward at each step is a linear function of present state and action.
        However, we can only observe aggregated reward at the last step

        This model will fail to learn when ground-truth reward is a function of
        multiple steps' states and actions.
        """
        state_dim = 10
        action_dim = 2
        seq_len = 5
        batch_size = 512
        num_batches = 5000
        sizes = [256, 128]
        activations = ["relu", "relu"]
        last_layer_activation = "linear"
        reward_net = SyntheticRewardNet(
            SingleStepSyntheticRewardNet(
                state_dim=state_dim,
                action_dim=action_dim,
                sizes=sizes,
                activations=activations,
                last_layer_activation=last_layer_activation,
            )
        )
        optimizer = Optimizer__Union(Adam=classes["Adam"]())
        trainer = RewardNetTrainer(reward_net, optimizer)
        trainer.set_reporter(
            RewardNetworkReporter(
                trainer.loss_type,
                str(reward_net),
            )
        )
        if ground_truth_reward_from_multiple_steps:
            weight, data = create_sequence_data(
                state_dim, action_dim, seq_len, batch_size, num_batches
            )
        else:
            weight, data = create_data(
                state_dim, action_dim, seq_len, batch_size, num_batches
            )
        avg_eval_loss = train_and_eval(trainer, data)
        return avg_eval_loss

    def test_single_step_parametric_binary_reward(self):
        """
        Reward at each step is a linear function of present state and action.
        However, we can only observe aggregated reward at the last step

        This model will fail to learn when ground-truth reward is a function of
        multiple steps' states and actions.
        """
        state_dim = 10
        action_dim = 2
        seq_len = 5
        batch_size = 512
        num_batches = 5000
        sizes = [256, 128]
        activations = ["relu", "relu"]
        last_layer_activation = "sigmoid"
        reward_net = SyntheticRewardNet(
            SingleStepSyntheticRewardNet(
                state_dim=state_dim,
                action_dim=action_dim,
                sizes=sizes,
                activations=activations,
                last_layer_activation=last_layer_activation,
            )
        )
        optimizer = Optimizer__Union(Adam=classes["Adam"]())
        trainer = RewardNetTrainer(
            reward_net, optimizer, loss_type=LossFunction.BCELoss
        )
        trainer.set_reporter(
            RewardNetworkReporter(
                trainer.loss_type,
                str(reward_net),
            )
        )
        weight, data = create_data(
            state_dim, action_dim, seq_len, batch_size, num_batches, binary_reward=True
        )
        avg_eval_loss = train_and_eval(trainer, data)
        return avg_eval_loss

    def test_ngram_fc_parametric_reward(self):
        """
        Reward at each step is a linear function of states and actions in a
        context window around the step.

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
        reward_net = SyntheticRewardNet(
            NGramFullyConnectedNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                sizes=sizes,
                activations=activations,
                last_layer_activation=last_layer_activation,
                context_size=3,
            )
        )
        optimizer = Optimizer__Union(Adam=classes["Adam"]())
        trainer = RewardNetTrainer(reward_net, optimizer)
        trainer.set_reporter(
            RewardNetworkReporter(
                trainer.loss_type,
                str(reward_net),
            )
        )
        weight, data = create_sequence_data(
            state_dim, action_dim, seq_len, batch_size, num_batches
        )
        threshold = 0.2
        avg_eval_loss = train_and_eval(trainer, data)
        assert avg_eval_loss < threshold

    def test_ngram_conv_net_parametric_reward(self):
        """
        Reward at each step is a linear function of states and actions in a
        context window around the step.

        However, we can only observe aggregated reward at the last step
        """
        state_dim = 10
        action_dim = 2
        seq_len = 5
        batch_size = 512
        num_batches = 5000
        sizes = [128, 64]
        activations = ["relu", "relu"]
        last_layer_activation = "linear"
        conv_net_params = rlp.ConvNetParameters(
            conv_dims=[128],
            conv_height_kernels=[1],
            pool_types=["max"],
            pool_kernel_sizes=[1],
        )
        reward_net = SyntheticRewardNet(
            NGramConvolutionalNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                sizes=sizes,
                activations=activations,
                last_layer_activation=last_layer_activation,
                context_size=3,
                conv_net_params=conv_net_params,
            )
        )
        optimizer = Optimizer__Union(Adam=classes["Adam"]())
        trainer = RewardNetTrainer(reward_net, optimizer)
        trainer.set_reporter(
            RewardNetworkReporter(
                trainer.loss_type,
                str(reward_net),
            )
        )
        weight, data = create_sequence_data(
            state_dim, action_dim, seq_len, batch_size, num_batches
        )
        threshold = 0.2
        avg_eval_loss = train_and_eval(trainer, data)
        assert avg_eval_loss < threshold, "loss = {} larger than threshold {}".format(
            avg_eval_loss, threshold
        )

    def test_lstm_parametric_reward(self):
        """
        Reward at each step is a linear function of states and actions in a
        context window around the step.

        However, we can only observe aggregated reward at the last step
        """
        state_dim = 10
        action_dim = 2
        seq_len = 5
        batch_size = 512
        num_batches = 5000
        last_layer_activation = "linear"
        reward_net = SyntheticRewardNet(
            SequenceSyntheticRewardNet(
                state_dim=state_dim,
                action_dim=action_dim,
                lstm_hidden_size=128,
                lstm_num_layers=2,
                lstm_bidirectional=True,
                last_layer_activation=last_layer_activation,
            )
        )
        optimizer = Optimizer__Union(Adam=classes["Adam"]())
        trainer = RewardNetTrainer(reward_net, optimizer)
        trainer.set_reporter(
            RewardNetworkReporter(
                trainer.loss_type,
                str(reward_net),
            )
        )
        weight, data = create_sequence_data(
            state_dim, action_dim, seq_len, batch_size, num_batches
        )
        threshold = 0.2
        avg_eval_loss = train_and_eval(trainer, data)
        assert avg_eval_loss < threshold

    def test_transformer_parametric_reward(self):
        """
        Reward at each step is a linear function of states and actions in a
        context window around the step.

        However, we can only observe aggregated reward at the last step
        """
        state_dim = 10
        action_dim = 2
        seq_len = 5
        batch_size = 512
        num_batches = 10000
        d_model = 64
        nhead = 8
        num_encoder_layers = 1
        dim_feedforward = 64
        last_layer_activation = "linear"
        max_len = seq_len + 1
        reward_net = SyntheticRewardNet(
            TransformerSyntheticRewardNet(
                state_dim=state_dim,
                action_dim=action_dim,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=0.0,
                activation="relu",
                last_layer_activation=last_layer_activation,
                layer_norm_eps=1e-5,
                max_len=max_len,
            )
        )
        optimizer = Optimizer__Union(Adam=classes["Adam"]())
        trainer = RewardNetTrainer(reward_net, optimizer)
        trainer.set_reporter(
            RewardNetworkReporter(
                trainer.loss_type,
                str(reward_net),
            )
        )
        weight, data = create_sequence_data(
            state_dim, action_dim, seq_len, batch_size, num_batches
        )

        threshold = 0.25
        avg_eval_loss = train_and_eval(trainer, data)
        assert (
            avg_eval_loss < threshold
        ), "loss = {:.4f} larger than threshold {}".format(avg_eval_loss, threshold)
