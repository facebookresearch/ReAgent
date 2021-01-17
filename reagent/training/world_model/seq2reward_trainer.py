#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging

import reagent.types as rlt
import torch
import torch.nn as nn
import torch.nn.functional as F
from reagent.core.tracker import observable
from reagent.models.fully_connected_network import FullyConnectedNetwork
from reagent.models.seq2reward_model import Seq2RewardNetwork
from reagent.parameters import Seq2RewardTrainerParameters
from reagent.training.loss_reporter import NoOpLossReporter
from reagent.training.trainer import Trainer
from reagent.training.utils import gen_permutations

logger = logging.getLogger(__name__)


# pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
#  its type `no_grad` is not callable.
@torch.no_grad()
def get_step_prediction(
    step_predict_network: FullyConnectedNetwork, training_batch: rlt.MemoryNetworkInput
):
    first_step_state = training_batch.state.float_features[0]
    pred_reward_len_output = step_predict_network(first_step_state)
    step_probability = F.softmax(pred_reward_len_output, dim=1)
    return step_probability


# pyre-fixme[56]: Decorator `torch.no_grad(...)` could not be called, because
#  its type `no_grad` is not callable.
@torch.no_grad()
def get_Q(
    seq2reward_network: Seq2RewardNetwork,
    cur_state: torch.Tensor,
    all_permut: torch.Tensor,
) -> torch.Tensor:
    """
    Input:
        cur_state: the current state from where we start planning.
            shape: batch_size x state_dim
        all_permut: all action sequences (sorted in lexical order) for enumeration
            shape: seq_len x num_perm x action_dim
    """
    batch_size = cur_state.shape[0]
    _, num_permut, num_action = all_permut.shape
    num_permut_per_action = int(num_permut / num_action)

    preprocessed_state = cur_state.unsqueeze(0).repeat_interleave(num_permut, dim=1)
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


@observable(
    mse_loss=torch.Tensor, step_entropy_loss=torch.Tensor, q_values=torch.Tensor
)
class Seq2RewardTrainer(Trainer):
    """ Trainer for Seq2Reward """

    def __init__(
        self, seq2reward_network: Seq2RewardNetwork, params: Seq2RewardTrainerParameters
    ):
        self.seq2reward_network = seq2reward_network
        self.params = params
        self.mse_optimizer = torch.optim.Adam(
            self.seq2reward_network.parameters(), lr=params.learning_rate
        )
        self.minibatch_size = self.params.batch_size
        self.loss_reporter = NoOpLossReporter()

        # PageHandler must use this to activate evaluator:
        self.calc_cpe_in_training = True
        # Turning off Q value output during training:
        self.view_q_value = params.view_q_value
        # permutations used to do planning
        self.all_permut = gen_permutations(
            params.multi_steps, len(self.params.action_names)
        )
        self.mse_loss = nn.MSELoss(reduction="mean")

        # Predict how many steps are remaining from the current step
        self.step_predict_network = FullyConnectedNetwork(
            [
                self.seq2reward_network.state_dim,
                self.params.step_predict_net_size,
                self.params.step_predict_net_size,
                self.params.multi_steps,
            ],
            ["relu", "relu", "linear"],
            use_layer_norm=False,
        )
        self.step_loss = nn.CrossEntropyLoss(reduction="mean")
        self.step_optimizer = torch.optim.Adam(
            self.step_predict_network.parameters(), lr=params.learning_rate
        )

    def train(self, training_batch: rlt.MemoryNetworkInput):
        mse_loss, step_entropy_loss = self.get_loss(training_batch)

        self.mse_optimizer.zero_grad()
        mse_loss.backward()
        self.mse_optimizer.step()

        self.step_optimizer.zero_grad()
        step_entropy_loss.backward()
        self.step_optimizer.step()

        detached_mse_loss = mse_loss.cpu().detach().item()
        detached_step_entropy_loss = step_entropy_loss.cpu().detach().item()

        if self.view_q_value:
            state_first_step = training_batch.state.float_features[0]
            q_values = (
                get_Q(
                    self.seq2reward_network,
                    state_first_step,
                    self.all_permut,
                )
                .cpu()
                .mean(0)
                .tolist()
            )
        else:
            q_values = [0] * len(self.params.action_names)

        step_probability = (
            get_step_prediction(self.step_predict_network, training_batch)
            .cpu()
            .mean(dim=0)
            .numpy()
        )
        logger.info(
            f"Seq2Reward trainer output: mse_loss={detached_mse_loss}, "
            f"step_entropy_loss={detached_step_entropy_loss}, q_values={q_values}, "
            f"step_probability={step_probability}"
        )
        # pyre-fixme[16]: `Seq2RewardTrainer` has no attribute `notify_observers`.
        self.notify_observers(
            mse_loss=detached_mse_loss,
            step_entropy_loss=detached_step_entropy_loss,
            q_values=[q_values],
        )
        return (detached_mse_loss, detached_step_entropy_loss, q_values)

    def get_loss(self, training_batch: rlt.MemoryNetworkInput):
        """
        Compute losses:
            MSE(predicted_acc_reward, target_acc_reward)

        :param training_batch:
            training_batch has these fields:
            - state: (SEQ_LEN, BATCH_SIZE, STATE_DIM) torch tensor
            - action: (SEQ_LEN, BATCH_SIZE, ACTION_DIM) torch tensor
            - reward: (SEQ_LEN, BATCH_SIZE) torch tensor

        :returns:
            mse loss on reward
            step_entropy_loss on step prediction
        """
        # pyre-fixme[16]: Optional type has no attribute `flatten`.
        valid_reward_len = training_batch.valid_next_seq_len.flatten()

        first_step_state = training_batch.state.float_features[0]
        valid_reward_len_output = self.step_predict_network(first_step_state)
        step_entropy_loss = self.step_loss(
            valid_reward_len_output, valid_reward_len - 1
        )

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
        mse = self.mse_loss(predicted_acc_reward, target_acc_reward)
        return mse, step_entropy_loss

    def warm_start_components(self):
        components = ["seq2reward_network"]
        return components
