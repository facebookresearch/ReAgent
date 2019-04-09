#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
import torch
import torch.nn as nn


class SimulatedWorldModel(nn.Module):
    """
    A world model used for simulation. Underlying is an RNN with fixed
    parameters. Given a sequence of actions and states, it outputs the next
    state's mixture means and reward.
    """

    def __init__(
        self,
        action_dim,
        state_dim,
        num_gaussians,
        lstm_num_hidden_layers,
        lstm_num_hiddens,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_gaussians = num_gaussians
        self.lstm_num_hidden_layers = lstm_num_hidden_layers
        self.lstm_num_hiddens = lstm_num_hiddens
        self.init_lstm()
        self.init_weight()
        self.init_hidden()
        self.eval()

    def init_lstm(self):
        self.lstm = nn.LSTM(
            input_size=self.action_dim + self.state_dim,
            hidden_size=self.lstm_num_hiddens,
            num_layers=self.lstm_num_hidden_layers,
        )
        # output mu for each guassian, and reward
        self.gmm_linear = nn.Linear(
            self.lstm_num_hiddens, self.state_dim * self.num_gaussians + 1
        )

    def init_hidden(self, batch_size=1):
        # (num_layers * num_directions, batch, hidden_size)
        self.hidden = (
            torch.zeros(self.lstm_num_hidden_layers, batch_size, self.lstm_num_hiddens),
            torch.zeros(self.lstm_num_hidden_layers, batch_size, self.lstm_num_hiddens),
        )

    def init_weight(self):
        torch.manual_seed(3212)
        for _, p in self.lstm.named_parameters():
            nn.init.normal_(p, 0, 1)
        for _, p in self.gmm_linear.named_parameters():
            nn.init.normal_(p, 0, 1)

    def forward(self, actions, cur_states):
        # actions: (SEQ_LEN, BATCH_SIZE, ACTION_SIZE)
        # cur_states: (SEQ_LEN, BATCH_SIZE, FEATURE_SIZE)
        seq_len, batch_size = actions.size(0), actions.size(1)

        X = torch.cat([actions, cur_states], dim=-1)
        # X_shape: (1, act_seq_len, lstm_input_dim)
        Y, self.hidden = self.lstm(X, self.hidden)
        gmm_outs = self.gmm_linear(Y)

        mus = gmm_outs[:, :, :-1]
        mus = mus.view(seq_len, batch_size, self.num_gaussians, self.state_dim)
        rewards = gmm_outs[:, :, -1]

        return mus, rewards
