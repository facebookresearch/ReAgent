#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DDPGPredictor(object):
    def __init__(self, trainer) -> None:
        self.actor = ActorNet(trainer.env_details.action_range,
                trainer.actor_params.layers, trainer.actor_params.activations)

    def predict(self, states) -> List[List[float]]:
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
        """ Forward pass for actor network
        :param state state as list of state features
        """
        x = torch.from_numpy(state).float()
        for i, activation in enumerate(self.activations):
            activation_func = getattr(F, activation)
            fc_func = getattr(self, self.layer_names[i])
            x = activation_func(fc_func(x))

        scale_tensor = torch.from_numpy(self.action_range_high).float()
        # Scale output to be in [action_range_low, action_range_high]
        return torch.matmul(x, scale_tensor)
