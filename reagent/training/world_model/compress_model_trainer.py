#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.models.fully_connected_network import FullyConnectedNetwork
from reagent.models.seq2reward_model import Seq2RewardNetwork
from reagent.parameters import Seq2RewardTrainerParameters
from reagent.training.loss_reporter import NoOpLossReporter
from reagent.training.trainer import Trainer
from reagent.training.utils import gen_permutations


logger = logging.getLogger(__name__)


class CompressModelTrainer(Trainer):
    """ Trainer for Seq2Reward """

    def __init__(
        self,
        compress_model_network: FullyConnectedNetwork,
        seq2reward_network: Seq2RewardNetwork,
        params: Seq2RewardTrainerParameters,
    ):
        self.compress_model_network = compress_model_network
        self.seq2reward_network = seq2reward_network
        self.params = params
        self.optimizer = torch.optim.Adam(
            self.compress_model_network.parameters(), lr=params.learning_rate
        )
        self.minibatch_size = self.params.batch_size
        self.loss_reporter = NoOpLossReporter()

        # PageHandler must use this to activate evaluator:
        self.calc_cpe_in_training = True

    def train(self, training_batch: rlt.MemoryNetworkInput):
        self.optimizer.zero_grad()
        loss = self.get_loss(training_batch)
        loss.backward()
        self.optimizer.step()
        detached_loss = loss.cpu().detach().item()

        return detached_loss

    def get_loss(self, training_batch: rlt.MemoryNetworkInput):
        compress_model_output = self.compress_model_network(
            training_batch.state.float_features[0]
        )
        target = self.get_Q(
            training_batch,
            training_batch.batch_size(),
            self.params.multi_steps,
            len(self.params.action_names),
        )
        assert (
            compress_model_output.size() == target.size()
        ), f"{compress_model_output.size()}!={target.size()}"
        mse = F.mse_loss(compress_model_output, target)
        return mse

    def warm_start_components(self):
        logger.info("No warm start components yet...")
        components = []
        return components

    # pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
    #  its type `no_grad` is not callable.
    @torch.no_grad()
    def get_Q(
        self,
        batch: rlt.MemoryNetworkInput,
        batch_size: int,
        seq_len: int,
        num_action: int,
    ) -> torch.Tensor:
        try:
            # pyre-fixme[16]: `Seq2RewardTrainer` has no attribute `all_permut`.
            self.all_permut
        except AttributeError:
            self.all_permut = gen_permutations(seq_len, num_action)
            # pyre-fixme[16]: `Seq2RewardTrainer` has no attribute `num_permut`.
            self.num_permut = self.all_permut.size(1)

        preprocessed_state = (
            batch.state.float_features[0]
            .unsqueeze(0)
            .repeat_interleave(self.num_permut, dim=1)
        )
        state_feature_vector = rlt.FeatureData(preprocessed_state)

        # expand action to match the expanded state sequence
        action = self.all_permut.repeat(1, batch_size, 1)
        # state_feature_vector: [1, BATCH_SIZE * NUM_PERMUT, STATE_DIM]
        # action: [SEQ_LEN, BATCH_SIZE * NUM_PERMUT, ACTION_DIM]
        # acc_reward: [BATCH_SIZE * NUM_PERMUT, 1]
        reward = self.seq2reward_network(
            state_feature_vector, rlt.FeatureData(action)
        ).acc_reward.reshape(batch_size, num_action, self.num_permut // num_action)

        # The permuations are generated with lexical order
        # the output has shape [num_perm, num_action,1]
        # that means we can aggregate on the max reward
        # then reshape it to (BATCH_SIZE, ACT_DIM)
        max_reward = (
            # pyre-fixme[16]: `Tuple` has no attribute `values`.
            torch.max(reward, 2)
            .values.cpu()
            .detach()
            .reshape(batch_size, num_action)
        )

        return max_reward
