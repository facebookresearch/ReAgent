#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import logging
from collections import deque
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from ml.rl import types as rlt
from torch.distributions.normal import Normal


logger = logging.getLogger(__name__)


class _MDNRNNBase(nn.Module):
    def __init__(
        self, state_dim, action_dim, num_hiddens, num_hidden_layers, num_gaussians
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_hiddens = num_hiddens
        self.num_hidden_layers = num_hidden_layers
        self.num_gaussians = num_gaussians

        # outputs:
        # 1. mu, sigma, and pi for each gaussian
        # 2. non-terminal signal
        # 3. reward
        self.gmm_linear = nn.Linear(
            num_hiddens, (2 * state_dim + 1) * num_gaussians + 2
        )

    def forward(self, *inputs):
        pass


class MDNRNN(_MDNRNNBase):
    """ Mixture Density Network - Recurrent Neural Network """

    def __init__(
        self, state_dim, action_dim, num_hiddens, num_hidden_layers, num_gaussians
    ):
        super().__init__(
            state_dim, action_dim, num_hiddens, num_hidden_layers, num_gaussians
        )
        self.rnn = nn.LSTM(
            input_size=state_dim + action_dim,
            hidden_size=num_hiddens,
            num_layers=num_hidden_layers,
        )

    def forward(self, actions, states, hidden=None):
        """ Forward pass of MDN-RNN

        :param actions: (SEQ_LEN, BATCH_SIZE, ACTION_DIM) torch tensor
        :param states: (SEQ_LEN, BATCH_SIZE, STATE_DIM) torch tensor

        :returns: parameters of the GMM prediction for the next state,
        gaussian prediction of the reward and logit prediction of
        non-terminality.
            - mus: (SEQ_LEN, BATCH_SIZE, NUM_GAUSSIANS, STATE_DIM) torch tensor
            - sigmas: (SEQ_LEN, BATCH_SIZE, NUM_GAUSSIANS, STATE_DIM) torch tensor
            - logpi: (SEQ_LEN, BATCH_SIZE, NUM_GAUSSIANS) torch tensor
            - reward: (SEQ_LEN, BATCH_SIZE) torch tensor
            - not_terminal: (SEQ_LEN, BATCH_SIZE) torch tensor
        """
        seq_len, batch_size = actions.size(0), actions.size(1)
        ins = torch.cat([actions, states], dim=-1)
        outs, new_hidden = self.rnn(ins, hidden)
        gmm_outs = self.gmm_linear(outs)

        stride = self.num_gaussians * self.state_dim

        mus = gmm_outs[:, :, :stride].view(
            seq_len, batch_size, self.num_gaussians, self.state_dim
        )
        sigmas = torch.exp(
            gmm_outs[:, :, stride : 2 * stride].view(
                seq_len, batch_size, self.num_gaussians, self.state_dim
            )
        )
        logpi = f.log_softmax(
            gmm_outs[:, :, 2 * stride : 2 * stride + self.num_gaussians].view(
                seq_len, batch_size, self.num_gaussians
            ),
            dim=-1,
        )
        reward = gmm_outs[:, :, -2]
        not_terminal = gmm_outs[:, :, -1]

        return mus, sigmas, logpi, reward, not_terminal, new_hidden

    def get_initial_hidden_state(self, batch_size=1):
        hidden = (
            torch.zeros(self.num_hidden_layers, batch_size, self.num_hiddens),
            torch.zeros(self.num_hidden_layers, batch_size, self.num_hiddens),
        )
        return hidden


class MDNRNNMemorySample(NamedTuple):
    state: np.array
    action: np.array
    next_state: np.array
    reward: float
    not_terminal: float


class MDNRNNMemoryPool:
    def __init__(self, max_replay_memory_size):
        self.replay_memory = deque(maxlen=max_replay_memory_size)
        self.max_replay_memory_size = max_replay_memory_size
        self.accu_memory_num = 0

    def deque_sample(self, indices):
        for i in indices:
            s = self.replay_memory[i]
            yield s.state, s.action, s.next_state, s.reward, s.not_terminal

    def sample_memories(self, batch_size, use_gpu=False, batch_first=False):
        """
        :param batch_size: number of samples to return
        :param use_gpu: whether to put samples on gpu
        :param batch_first: If True, the first dimension of data is batch_size.
            If False (default), the first dimension is SEQ_LEN. Therefore,
            state's shape is SEQ_LEN x BATCH_SIZE x STATE_DIM, for example. By default,
            MDN-RNN consumes data with SEQ_LEN as the first dimension.
        """
        sample_indices = np.random.randint(self.memory_size, size=batch_size)
        device = torch.device("cuda") if use_gpu else torch.device("cpu")
        # state/next state shape: batch_size x seq_len x state_dim
        # action shape: # state shape: batch_size x seq_len x action_dim
        # reward/not_terminal shape: batch_size x seq_len
        state, action, next_state, reward, not_terminal = map(
            lambda x: torch.tensor(x, dtype=torch.float, device=device),
            zip(*self.deque_sample(sample_indices)),
        )

        if not batch_first:
            state, action, next_state, reward, not_terminal = transpose(
                state, action, next_state, reward, not_terminal
            )

        training_input = rlt.MemoryNetworkInput(
            state=rlt.FeatureVector(float_features=state),
            action=rlt.FeatureVector(float_features=action),
            next_state=next_state,
            reward=reward,
            not_terminal=not_terminal,
        )
        return rlt.TrainingBatch(training_input=training_input, extras=None)

    def insert_into_memory(self, state, action, next_state, reward, not_terminal):
        self.replay_memory.append(
            MDNRNNMemorySample(
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                not_terminal=not_terminal,
            )
        )
        self.accu_memory_num += 1

    @property
    def memory_size(self):
        return min(self.accu_memory_num, self.max_replay_memory_size)


def transpose(*args):
    res = []
    for arg in args:
        res.append(arg.transpose(1, 0))
    return res


def gmm_loss(batch, mus, sigmas, logpi, reduce=True):
    """ Computes the gmm loss.

    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.

    :param batch: (bs1, bs2, *, fs) torch tensor
    :param mus: (bs1, bs2, *, gs, fs) torch tensor
    :param sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :param logpi: (bs1, bs2, *, gs) torch tensor
    :param reduce: if not reduce, the mean in the following formula is omitted

    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))

    NOTE: The loss is not reduced along the feature dimension (i.e. it should
    scale linearily with fs).

    Adapted from: https://github.com/ctallec/world-models
    """
    # for non-image based environment, batch's shape before unsqueeze:
    # (seq_len, batch_size, fs)
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    # According to the world model paper, the prediction of next state is a
    # factored Gaussian distribution (i.e., the covariance matrix of the multi-
    # dimensional Gaussian distribution is a diagonal matrix). Hence we can sum
    # log probability of each dimension when calculating log joint probability
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    # log sum exp
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return -torch.mean(log_prob)
    return -log_prob
