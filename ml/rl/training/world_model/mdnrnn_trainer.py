#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from collections import deque
from typing import Deque, Optional

import ml.rl.types as rlt
import torch
import torch.nn.functional as F
from ml.rl.models.mdn_rnn import gmm_loss, transpose
from ml.rl.models.world_model import MemoryNetwork
from ml.rl.thrift.core.ttypes import MDNRNNParameters


logger = logging.getLogger(__name__)


class MDNRNNTrainer:
    def __init__(
        self,
        mdnrnn_network: MemoryNetwork,
        params: MDNRNNParameters,
        cum_loss_hist=100,
        fit_only_one_next_step=False,
    ):
        self.mdnrnn = mdnrnn_network
        self.params = params
        self.optimizer = torch.optim.Adam(
            self.mdnrnn.mdnrnn.parameters(), lr=params.learning_rate
        )
        self.minibatch = 0
        self.cum_loss: Deque[float] = deque([], maxlen=cum_loss_hist)
        self.cum_bce: Deque[float] = deque([], maxlen=cum_loss_hist)
        self.cum_gmm: Deque[float] = deque([], maxlen=cum_loss_hist)
        self.cum_mse: Deque[float] = deque([], maxlen=cum_loss_hist)
        self.fit_only_one_next_step = fit_only_one_next_step

    def train(self, training_batch, batch_first=False):
        assert (
            type(training_batch) is rlt.TrainingBatch
            and type(training_batch.training_input) is rlt.MemoryNetworkInput
        )

        self.minibatch += 1
        if batch_first:
            batch_size, seq_len, state_dim = (
                training_batch.training_input.state.float_features.shape
            )
        else:
            seq_len, batch_size, state_dim = (
                training_batch.training_input.state.float_features.shape
            )

        self.mdnrnn.mdnrnn.train()
        self.optimizer.zero_grad()
        losses = self.get_loss(training_batch, state_dim, batch_first)
        losses["loss"].backward()
        self.optimizer.step()

        detached_losses = {
            "loss": losses["loss"].cpu().detach().item(),
            "gmm": losses["gmm"].cpu().detach().item(),
            "bce": losses["bce"].cpu().detach().item(),
            "mse": losses["mse"].cpu().detach().item(),
        }
        self.cum_loss += [detached_losses["loss"]]
        self.cum_gmm += [detached_losses["gmm"]]
        self.cum_bce += [detached_losses["bce"]]
        self.cum_mse += [detached_losses["mse"]]
        del losses

        return detached_losses

    def get_loss(
        self,
        training_batch: rlt.TrainingBatch,
        state_dim: Optional[int] = None,
        batch_first: bool = False,
    ):
        """
        Compute losses.

        The loss that is computed is:
            (GMMLoss(next_state, GMMPredicted) + MSE(reward, predicted_reward) +
            BCE(not_terminal, logit_not_terminal)) / (STATE_DIM + 2)

        The STATE_DIM + 2 factor is here to counteract the fact that the GMMLoss scales
            approximately linearily with STATE_DIM, the feature size of states. All losses
            are averaged both on the batch and the sequence dimensions (the two first
            dimensions).

        :param training_batch
            training_batch.learning_input has these fields:
            - state: (BATCH_SIZE, SEQ_LEN, STATE_DIM) torch tensor
            - action: (BATCH_SIZE, SEQ_LEN, ACTION_DIM) torch tensor
            - reward: (BATCH_SIZE, SEQ_LEN) torch tensor
            - not-terminal: (BATCH_SIZE, SEQ_LEN) torch tensor
            - next_state: (BATCH_SIZE, SEQ_LEN, STATE_DIM) torch tensor
            the first two dimensions may be swapped depending on batch_first

        :param state_dim: the dimension of states. If provided, use it to normalize
            gmm loss

        :param batch_first: whether data's first dimension represents batch size. If
            FALSE, state, action, reward, not-terminal, and next_state's first
            two dimensions are SEQ_LEN and BATCH_SIZE.

        :returns: dictionary of losses, containing the gmm, the mse, the bce and
            the averaged loss.
        """
        learning_input = training_batch.training_input
        # mdnrnn's input should have seq_len as the first dimension
        if batch_first:
            state, action, next_state, reward, not_terminal = transpose(
                learning_input.state.float_features,
                learning_input.action.float_features,
                learning_input.next_state,
                learning_input.reward,
                learning_input.not_terminal,
            )
            learning_input = rlt.MemoryNetworkInput(
                state=rlt.FeatureVector(float_features=state),
                action=rlt.FeatureVector(float_features=action),
                next_state=next_state,
                reward=reward,
                not_terminal=not_terminal,
            )

        mdnrnn_input = rlt.StateAction(
            state=learning_input.state, action=learning_input.action
        )
        mdnrnn_output = self.mdnrnn(mdnrnn_input)
        mus, sigmas, logpi, rs, nts = (
            mdnrnn_output.mus,
            mdnrnn_output.sigmas,
            mdnrnn_output.logpi,
            mdnrnn_output.reward,
            mdnrnn_output.not_terminal,
        )

        next_state = learning_input.next_state
        not_terminal = learning_input.not_terminal
        reward = learning_input.reward
        if self.fit_only_one_next_step:
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
