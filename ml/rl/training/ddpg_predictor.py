#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List


class DDPGPredictor(object):
    def __init__(self, trainer) -> None:
        self.actor = ActorNet(trainer.env_details.action_range,
            trainer.actor_params.layers, trainer.actor_params.activations)
        self.critic = CriticNet(trainer.critic_params.layers,
            trainer.critic_params.activations)

    def predict_action(self, states) -> List[List[float]]:
        """ Returns list of actions output from actor network
        :param states states as list of states to produce actions for
        """
        examples = []
        for state in states:
            example = np.zeros([len(state)])
            for k, v in state.items():
                example[k] = v
            examples.append(example)
        output = [self.actor(example) for example in examples]
        return output

    def predict_q_value(self, states, actions) -> List[float]:
        """ Returns list of q values from critic network for <state, action> inputs
        :param states states as list of state dicts
        :param actions actions as list of action dicts
        """
        examples = []
        for i in range(len(states)):
            state = np.zeros([len(states[i])])
            for k, v in states[i].items():
                state[k] = v
            action = np.zeros([len(actions[i])])
            for k, v in actions[i].items():
                action[k - len(state)] = v
            examples.append((state, action))
        output = [self.critic(example[0], example[1]) for example in examples]
        return output

    @classmethod
    def export_actor(cls, trainer):
        return DDPGPredictor(trainer)


class ActorNet(nn.Module):
    def __init__(self, action_range, layers, activations) -> None:
        super(ActorNet, self).__init__()
        self.layer_names: List[str] = []
        self.action_range_high = action_range[1]
        self.activations = activations

        assert (
            len(layers) >= 3
        ), 'Invalid layer schema {} for actor network'.format(layers)

        for i, layer in enumerate(layers[1:]):
            self.layer_names.append('fc{}'.format(i + 1))
            setattr(self, self.layer_names[i], nn.Linear(layers[i], layer))

    def forward(self, state) -> torch.FloatTensor:
        """ Forward pass for actor network. Assumes activation names are
        valid pytorch activation names.
        :param state state as list of state features
        """
        x = Variable(torch.from_numpy(state).float())
        for i, activation in enumerate(self.activations):
            activation_func = getattr(F, activation)
            fc_func = getattr(self, self.layer_names[i])
            x = fc_func(x) if activation == 'linear' else activation_func(fc_func(x))

        scale_tensor = torch.from_numpy(self.action_range_high).float()
        # Scale output to be in [action_range_low, action_range_high]
        return torch.matmul(x, scale_tensor)


class CriticNet(nn.Module):
    def __init__(self, layers, activations) -> None:
        super(CriticNet, self).__init__()
        self.layer_names: List[str] = []
        self.activations = activations

        assert (
            len(layers) >= 3
        ), 'Invalid layer schema {} for critic network'.format(layers)

        for i, layer in enumerate(layers[1:]):
            self.layer_names.append('fc{}'.format(i + 1))
            setattr(self, self.layer_names[i], nn.Linear(layers[i], layer))

    def forward(self, state, action) -> torch.FloatTensor:
        """ Forward pass for critic network. Assumes activation names are
        valid pytorch activation names.
        :param state state as list of state features
        :param state action as list of action features
        """
        x = torch.cat(
            (Variable(torch.from_numpy(state).float()),
            Variable(torch.from_numpy(action).float())), dim=0
        )
        for i, activation in enumerate(self.activations):
            activation_func = getattr(F, activation)
            fc_func = getattr(self, self.layer_names[i])
            x = fc_func(x) if activation == 'linear' else activation_func(fc_func(x))
        return x
