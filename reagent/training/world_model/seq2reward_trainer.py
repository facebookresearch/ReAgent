#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.models.seq2reward_model import Seq2RewardNetwork
from reagent.parameters import Seq2RewardTrainerParameters
from reagent.training.loss_reporter import NoOpLossReporter
from reagent.training.trainer import Trainer


logger = logging.getLogger(__name__)


class Seq2RewardTrainer(Trainer):
    """ Trainer for Seq2Reward """

    def __init__(
        self, seq2reward_network: Seq2RewardNetwork, params: Seq2RewardTrainerParameters
    ):
        self.seq2reward_network = seq2reward_network
        self.params = params
        self.optimizer = torch.optim.Adam(
            self.seq2reward_network.parameters(), lr=params.learning_rate
        )
        self.minibatch_size = self.params.batch_size
        self.loss_reporter = NoOpLossReporter()

        # PageHandler must use this to activate evaluator:
        self.calc_cpe_in_training = self.params.calc_cpe_in_training

    def train(self, training_batch: rlt.MemoryNetworkInput):
        self.optimizer.zero_grad()
        loss = self.get_loss(training_batch)
        loss.backward()
        self.optimizer.step()
        detached_loss = loss.cpu().detach().item()

        return detached_loss

    def get_loss(self, training_batch: rlt.MemoryNetworkInput):
        """
        Compute losses:
            MSE(predicted_acc_reward, target_acc_reward)

        :param training_batch:
            training_batch has these fields:
            - state: (SEQ_LEN, BATCH_SIZE, STATE_DIM) torch tensor
            - action: (SEQ_LEN, BATCH_SIZE, ACTION_DIM) torch tensor
            - reward: (SEQ_LEN, BATCH_SIZE) torch tensor

        :returns: mse loss on reward
        """

        seq2reward_output = self.seq2reward_network(
            training_batch.state, rlt.FeatureData(training_batch.action)
        )

        predicted_acc_reward = seq2reward_output.acc_reward
        target_rewards = training_batch.reward
        target_acc_reward = torch.sum(target_rewards, 0).unsqueeze(1)
        # make sure the prediction and target tensors have the same size
        # the size should both be (BATCH_SIZE, 1) in this case.
        assert predicted_acc_reward.size() == target_acc_reward.size()
        mse = F.mse_loss(predicted_acc_reward, target_acc_reward)
        return mse

    def warm_start_components(self):
        logger.info("No warm start components yet...")
        components = []
        return components
