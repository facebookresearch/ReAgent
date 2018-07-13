#!/usr/bin/env python3

import math

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from ml.rl.thrift.core.ttypes import AdditionalFeatureTypes

import logging

logger = logging.getLogger(__name__)

DEFAULT_ADDITIONAL_FEATURE_TYPES = AdditionalFeatureTypes(int_features=False)


class RLTrainer:
    # Q-value for action that is not possible. Guaranteed to be worse than any
    # legitimate action
    ACTION_NOT_POSSIBLE_VAL = -1e9

    def __init__(self, parameters, use_gpu, additional_feature_types):
        self.minibatch = 0
        self.reward_burnin = parameters.rl.reward_burnin
        self._additional_feature_types = additional_feature_types
        self.rl_temperature = parameters.rl.temperature
        self.gamma = parameters.rl.gamma
        self.tau = parameters.rl.target_update_rate
        self.use_seq_num_diff_as_time_diff = parameters.rl.use_seq_num_diff_as_time_diff

        if use_gpu and torch.cuda.is_available():
            logger.info("Using GPU: GPU requested and available.")
            self.use_gpu = True
            self.dtype = torch.cuda.FloatTensor
            self.dtypelong = torch.cuda.LongTensor
        else:
            logger.info("NOT Using GPU: GPU not requested or not available.")
            self.use_gpu = False
            self.dtype = torch.FloatTensor
            self.dtypelong = torch.LongTensor

    def _set_optimizer(self, optimizer_name):
        if optimizer_name == "ADAM":
            self.optimizer_func = torch.optim.Adam
        elif optimizer_name == "SGD":
            self.optimizer_func = torch.optim.SGD
        else:
            raise NotImplementedError(
                "{} optimizer not implemented".format(optimizer_name)
            )

    def _soft_update(self, network, target_network, tau) -> None:
        """ Target network update logic as defined in DDPG paper
        updated_params = tau * network_params + (1 - tau) * target_network_params
        :param network network with parameters to include in soft update
        :param target_network target network with params to soft update
        :param tau hyperparameter to control target tracking speed
        """
        for t_param, param in zip(target_network.parameters(), network.parameters()):
            new_param = tau * param.data + (1.0 - tau) * t_param.data
            t_param.data.copy_(new_param)

    def train(self, training_samples, evaluator=None, episode_values=None) -> None:
        raise NotImplementedError()

    def internal_prediction(self, input):
        """ Q-network forward pass method for internal domains.
        :param input input to network
        """
        self.q_network.eval()
        with torch.no_grad():
            input = Variable(torch.from_numpy(np.array(input)).type(self.dtype))
            q_values = self.q_network(input)
        self.q_network.train()
        return q_values.cpu().data.numpy()

    def get_value_from_timeline(self, discount_factor, reward_timeline):
        result = 0
        for time, reward in reward_timeline.items():
            result += (discount_factor ** time) * reward
        return result

    def internal_reward_estimation(self, input):
        """ Reward-network forward pass for internal domains. """
        self.reward_network.eval()
        with torch.no_grad():
            input = Variable(torch.from_numpy(np.array(input)).type(self.dtype))
            reward_estimates = self.reward_network(input)
        self.reward_network.train()
        return reward_estimates.cpu().data.numpy()


def guassian_fill_w_gain(tensor, activation, dim_in) -> None:
    """ Gaussian initialization with gain."""
    gain = math.sqrt(2) if activation == "relu" else 1
    init.normal_(tensor, mean=0, std=gain * math.sqrt(1 / dim_in))


class GenericFeedForwardNetwork(nn.Module):
    def __init__(self, layers, activations) -> None:
        super(GenericFeedForwardNetwork, self).__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        self.batch_norm_ops: nn.ModuleList = nn.ModuleList()
        self.activations = activations

        assert len(layers) >= 2, "Invalid layer schema {} for network".format(layers)

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            self.batch_norm_ops.append(nn.BatchNorm1d(layers[i]))
            guassian_fill_w_gain(self.layers[i].weight, self.activations[i], layers[i])
            init.constant_(self.layers[i].bias, 0)

    def forward(self, input) -> torch.FloatTensor:
        """ Forward pass for generic feed-forward DNNs. Assumes activation names
        are valid pytorch activation names.
        :param input tensor
        """
        if isinstance(input, np.ndarray):
            input = Variable(torch.from_numpy(input))

        x = input
        for i, activation in enumerate(self.activations):
            # TODO: (edoardoc) T30535967 Renable batchnorm when T30535876 is fixed
            # x = self.batch_norm_ops[i](x)
            activation_func = getattr(F, activation)
            fc_func = self.layers[i]
            x = fc_func(x) if activation == "linear" else activation_func(fc_func(x))
        return x
