#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
from reagent import types as rlt
from reagent.models.base import ModelBase


class Seq2RewardNetwork(ModelBase):
    def __init__(self, state_dim, action_dim, num_hiddens, num_hidden_layers):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_hiddens = num_hiddens
        self.num_hidden_layers = num_hidden_layers
        self.rnn = nn.LSTM(
            input_size=action_dim, hidden_size=num_hiddens, num_layers=num_hidden_layers
        )

        self.lstm_linear = nn.Linear(num_hiddens, 1)
        self.map_linear = nn.Linear(state_dim, self.num_hiddens)

    def input_prototype(self):
        return (
            rlt.FeatureData(torch.randn(1, 1, self.state_dim)),
            rlt.FeatureData(torch.randn(1, 1, self.action_dim)),
        )

    def forward(self, state: rlt.FeatureData, action: rlt.FeatureData):
        """ Forward pass of Seq2Reward

        Takes in the current state and use it as init hidden
        The input sequence are pure actions only
        Output the predicted reward after each time step

        :param actions: (SEQ_LEN, BATCH_SIZE, ACTION_DIM) torch tensor
        :param states: (SEQ_LEN, BATCH_SIZE, STATE_DIM) torch tensor

        :returns: predicated accumulated rewards at last step for the given sequence
            - reward: (BATCH_SIZE, 1) torch tensor
        """
        states = state.float_features
        actions = action.float_features
        hidden = self.get_initial_hidden_state(
            states[0][None, :, :], batch_size=states.size(1)
        )
        # use last hidden from the topmost hidden layer to predict reward
        # the size of reward should be converted to (BATCH_SIZE, 1)
        all_steps_hidden, last_step_hidden_and_cell = self.rnn(actions, hidden)
        lstm_outs = self.lstm_linear(last_step_hidden_and_cell[0])
        reward = lstm_outs[-1, :, -1].unsqueeze(1)

        return rlt.Seq2RewardOutput(acc_reward=reward)

    def get_initial_hidden_state(self, state, batch_size=1):
        # state embedding with linear mapping
        # repeat state to fill num_hidden_layers at first dimension
        state = state.repeat(self.num_hidden_layers, 1, 1)
        state_embed = self.map_linear(state)

        # hidden = (hidden,cell) where hidden is init with liner map
        # of input state and cell is 0.
        # hidden :
        # TUPLE(
        #     (NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE),
        #     (NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)
        # ) torch tensor
        hidden = (
            state_embed,
            torch.zeros(self.num_hidden_layers, batch_size, self.num_hiddens),
        )

        return hidden
