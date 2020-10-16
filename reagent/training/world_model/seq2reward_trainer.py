#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.models.seq2reward_model import Seq2RewardNetwork
from reagent.parameters import Seq2RewardTrainerParameters
from reagent.torch_utils import get_device
from reagent.training.loss_reporter import NoOpLossReporter
from reagent.training.trainer import Trainer
from reagent.training.utils import gen_permutations


logger = logging.getLogger(__name__)


# pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
#  its type `no_grad` is not callable.
@torch.no_grad()
def get_Q(
    seq2reward_network: Seq2RewardNetwork,
    batch: rlt.MemoryNetworkInput,
    all_permut: torch.Tensor,
) -> torch.Tensor:
    batch_size = batch.state.float_features.shape[1]
    _, num_permut, num_action = all_permut.shape
    num_permut_per_action = int(num_permut / num_action)

    preprocessed_state = (
        batch.state.float_features[0].unsqueeze(0).repeat_interleave(num_permut, dim=1)
    )
    state_feature_vector = rlt.FeatureData(preprocessed_state)

    # expand action to match the expanded state sequence
    action = rlt.FeatureData(all_permut.repeat(1, batch_size, 1))
    acc_reward = seq2reward_network(state_feature_vector, action).acc_reward.reshape(
        batch_size, num_action, num_permut_per_action
    )

    # The permuations are generated with lexical order
    # the output has shape [num_perm, num_action,1]
    # that means we can aggregate on the max reward
    # then reshape it to (BATCH_SIZE, ACT_DIM)
    max_acc_reward = (
        # pyre-fixme[16]: `Tuple` has no attribute `values`.
        torch.max(acc_reward, dim=2)
        .values.detach()
        .reshape(batch_size, num_action)
    )

    return max_acc_reward


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
        self.calc_cpe_in_training = True
        # Turning off Q value output during training:
        self.view_q_value = params.view_q_value
        # permutations used to do planning
        device = get_device(self.seq2reward_network)
        self.all_permut = gen_permutations(
            params.multi_steps, len(self.params.action_names)
        ).to(device)

    def train(self, training_batch: rlt.MemoryNetworkInput):
        self.optimizer.zero_grad()
        loss = self.get_loss(training_batch)
        loss.backward()
        self.optimizer.step()
        detached_loss = loss.cpu().detach().item()

        if self.view_q_value:
            q_values = (
                get_Q(self.seq2reward_network, training_batch, self.all_permut)
                .cpu()
                .mean(0)
                .tolist()
            )
        else:
            q_values = [0] * len(self.params.action_names)

        logger.info(f"Seq2Reward trainer output: {(detached_loss, q_values)}")
        return (detached_loss, q_values)

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
        # pyre-fixme[16]: Optional type has no attribute `flatten`.
        valid_reward_len = training_batch.valid_next_seq_len.flatten()

        seq2reward_output = self.seq2reward_network(
            training_batch.state,
            rlt.FeatureData(training_batch.action),
            valid_reward_len,
        )
        predicted_acc_reward = seq2reward_output.acc_reward

        seq_len, batch_size = training_batch.reward.size()
        gamma = self.params.gamma
        gamma_mask = (
            torch.Tensor(
                [[gamma ** i for i in range(seq_len)] for _ in range(batch_size)]
            )
            .transpose(0, 1)
            .to(training_batch.reward.device)
        )

        target_acc_rewards = torch.cumsum(training_batch.reward * gamma_mask, dim=0)
        target_acc_reward = target_acc_rewards[
            valid_reward_len - 1, torch.arange(batch_size)
        ].unsqueeze(1)

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
