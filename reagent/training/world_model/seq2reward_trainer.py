#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import reagent.core.types as rlt
import torch
import torch.nn.functional as F
from reagent.models.seq2reward_model import Seq2RewardNetwork
from reagent.parameters import Seq2RewardTrainerParameters
from reagent.reporting.world_model_reporter import WorldModelReporter
from reagent.training.trainer import Trainer
from reagent.training.utils import gen_permutations


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
        self.reporter = WorldModelReporter()

        # PageHandler must use this to activate evaluator:
        self.calc_cpe_in_training = True
        # Turning off Q value output during training:
        self.view_q_value = params.view_q_value

    def train(self, training_batch: rlt.MemoryNetworkInput):
        self.optimizer.zero_grad()
        loss = self.compute_loss(training_batch)
        loss.backward()
        self.optimizer.step()
        detached_loss = loss.cpu().detach().item()
        q_values = (
            self.get_Q(
                training_batch,
                training_batch.batch_size(),
                self.params.multi_steps,
                len(self.params.action_names),
            )
            .mean(0)
            .tolist()
        )
        self.reporter.report(mse=detached_loss)

        return (detached_loss, q_values)

    def compute_loss(self, training_batch: rlt.MemoryNetworkInput):
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
        seq_len, batch_size = target_rewards.size()
        gamma = self.params.gamma
        gamma_mask = torch.Tensor(
            [[gamma ** i for i in range(seq_len)] for _ in range(batch_size)]
        ).transpose(0, 1)
        target_acc_reward = torch.sum(target_rewards * gamma_mask, 0).unsqueeze(1)
        # make sure the prediction and target tensors have the same size
        # the size should both be (BATCH_SIZE, 1) in this case.
        assert (
            predicted_acc_reward.size() == target_acc_reward.size()
        ), f"{predicted_acc_reward.size()}!={target_acc_reward.size()}"
        mse = F.mse_loss(predicted_acc_reward, target_acc_reward)
        return mse

    def warm_start_components(self):
        components = ["seq2reward_network"]
        return components

    def get_Q(
        self,
        batch: rlt.MemoryNetworkInput,
        batch_size: int,
        seq_len: int,
        num_action: int,
    ) -> torch.Tensor:
        if not self.view_q_value:
            return torch.zeros(batch_size, num_action)
        try:
            # pyre-fixme[16]: `Seq2RewardTrainer` has no attribute `all_permut`.
            self.all_permut
        except AttributeError:
            self.all_permut = gen_permutations(seq_len, num_action)
            # pyre-fixme[16]: `Seq2RewardTrainer` has no attribute `num_permut`.
            self.num_permut = self.all_permut.size(1)

        preprocessed_state = batch.state.float_features.repeat_interleave(
            self.num_permut, dim=1
        )
        state_feature_vector = rlt.FeatureData(preprocessed_state)

        # expand action to match the expanded state sequence
        action = self.all_permut.repeat(1, batch_size, 1)
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
