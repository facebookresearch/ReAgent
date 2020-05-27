#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from collections import deque
from typing import Deque, Optional

import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.models.mdn_rnn import gmm_loss
from reagent.models.world_model import MemoryNetwork
from reagent.parameters import MDNRNNTrainerParameters
from reagent.training.trainer import Trainer


logger = logging.getLogger(__name__)


class MDNRNNTrainer(Trainer):
    """ Trainer for MDN-RNN """

    def __init__(
        self,
        memory_network: MemoryNetwork,
        params: MDNRNNTrainerParameters,
        cum_loss_hist: int = 100,
    ):
        self.memory_network = memory_network
        self.params = params
        self.optimizer = torch.optim.Adam(
            self.memory_network.mdnrnn.parameters(), lr=params.learning_rate
        )
        self.minibatch = 0
        self.cum_loss: Deque[float] = deque([], maxlen=cum_loss_hist)
        self.cum_bce: Deque[float] = deque([], maxlen=cum_loss_hist)
        self.cum_gmm: Deque[float] = deque([], maxlen=cum_loss_hist)
        self.cum_mse: Deque[float] = deque([], maxlen=cum_loss_hist)

    def train(self, training_batch: rlt.MemoryNetworkInput):
        self.minibatch += 1

        (seq_len, batch_size, state_dim) = training_batch.state.float_features.shape

        self.memory_network.mdnrnn.train()
        self.optimizer.zero_grad()
        losses = self.get_loss(training_batch, state_dim)
        losses["loss"].backward()
        self.optimizer.step()

        detached_losses = {k: loss.cpu().detach().item() for k, loss in losses.items()}
        self.cum_loss.append(detached_losses["loss"])
        self.cum_gmm.append(detached_losses["gmm"])
        self.cum_bce.append(detached_losses["bce"])
        self.cum_mse.append(detached_losses["mse"])
        del losses

        return detached_losses

    def get_loss(
        self, training_batch: rlt.MemoryNetworkInput, state_dim: Optional[int] = None
    ):
        """
        Compute losses:
            GMMLoss(next_state, GMMPredicted) / (STATE_DIM + 2)
            + MSE(reward, predicted_reward)
            + BCE(not_terminal, logit_not_terminal)

        The STATE_DIM + 2 factor is here to counteract the fact that the GMMLoss scales
            approximately linearly with STATE_DIM, dim of states. All losses
            are averaged both on the batch and the sequence dimensions (the two first
            dimensions).

        :param training_batch:
            training_batch has these fields:
            - state: (SEQ_LEN, BATCH_SIZE, STATE_DIM) torch tensor
            - action: (SEQ_LEN, BATCH_SIZE, ACTION_DIM) torch tensor
            - reward: (SEQ_LEN, BATCH_SIZE) torch tensor
            - not-terminal: (SEQ_LEN, BATCH_SIZE) torch tensor
            - next_state: (SEQ_LEN, BATCH_SIZE, STATE_DIM) torch tensor

        :param state_dim: the dimension of states. If provided, use it to normalize
            gmm loss

        :returns: dictionary of losses, containing the gmm, the mse, the bce and
            the averaged loss.
        """
        assert isinstance(training_batch, rlt.MemoryNetworkInput)
        # mdnrnn's input should have seq_len as the first dimension

        mdnrnn_output = self.memory_network(
            training_batch.state, rlt.FeatureData(training_batch.action)
        )
        # mus, sigmas: [seq_len, batch_size, num_gaussian, state_dim]
        mus, sigmas, logpi, rs, nts = (
            mdnrnn_output.mus,
            mdnrnn_output.sigmas,
            mdnrnn_output.logpi,
            mdnrnn_output.reward,
            mdnrnn_output.not_terminal,
        )

        next_state = training_batch.next_state.float_features
        not_terminal = training_batch.not_terminal
        reward = training_batch.reward
        if self.params.fit_only_one_next_step:
            next_state, not_terminal, reward, mus, sigmas, logpi, nts, rs = tuple(
                map(
                    lambda x: x[-1:],
                    (next_state, not_terminal, reward, mus, sigmas, logpi, nts, rs),
                )
            )

        gmm = (
            gmm_loss(next_state, mus, sigmas, logpi)
            * self.params.next_state_loss_weight
        )
        bce = (
            F.binary_cross_entropy_with_logits(nts, not_terminal)
            * self.params.not_terminal_loss_weight
        )
        mse = F.mse_loss(rs, reward) * self.params.reward_loss_weight
        if state_dim is not None:
            loss = gmm / (state_dim + 2) + bce + mse
        else:
            loss = gmm + bce + mse
        return {"gmm": gmm, "bce": bce, "mse": mse, "loss": loss}
