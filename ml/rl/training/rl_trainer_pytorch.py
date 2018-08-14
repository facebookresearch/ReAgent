#!/usr/bin/env python3

import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ml.rl.thrift.core.ttypes import AdditionalFeatureTypes
from torch.autograd import Variable


logger = logging.getLogger(__name__)

DEFAULT_ADDITIONAL_FEATURE_TYPES = AdditionalFeatureTypes(int_features=False)


class RLTrainer:
    # Q-value for action that is not possible. Guaranteed to be worse than any
    # legitimate action
    ACTION_NOT_POSSIBLE_VAL = -1e9
    # Hack to mark legitimate 0 value q-values before pytorch sparse -> dense
    FINGERPRINT = 12345

    def __init__(self, parameters, use_gpu, additional_feature_types, gradient_handler):
        self.minibatch = 0
        self.reward_burnin = parameters.rl.reward_burnin
        self._additional_feature_types = additional_feature_types
        self.rl_temperature = parameters.rl.temperature
        self.maxq_learning = parameters.rl.maxq_learning
        self.gamma = parameters.rl.gamma
        self.tau = parameters.rl.target_update_rate
        self.use_seq_num_diff_as_time_diff = parameters.rl.use_seq_num_diff_as_time_diff
        self.gradient_handler = gradient_handler

        if parameters.rl.q_network_loss == "mse":
            self.q_network_loss = getattr(F, "mse_loss")
        elif parameters.rl.q_network_loss == "huber":
            self.q_network_loss = getattr(F, "smooth_l1_loss")
        else:
            raise Exception(
                "Q-Network loss type {} not valid loss.".format(
                    parameters.rl.q_network_loss
                )
            )

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


def rescale_torch_tensor(
    tensor: torch.tensor,
    new_min: torch.tensor,
    new_max: torch.tensor,
    prev_min: torch.tensor,
    prev_max: torch.tensor,
):
    """
    Rescale column values in N X M torch tensor to be in new range.
    Each column m in input tensor will be rescaled from range
    [prev_min[m], prev_max[m]] to [new_min[m], new_max[m]]
    """
    assert tensor.shape[1] == new_min.shape[1] == new_max.shape[1]
    assert tensor.shape[1] == prev_min.shape[1] == prev_max.shape[1]
    prev_range = prev_max - prev_min
    new_range = new_max - new_min
    return ((tensor - prev_min) / prev_range) * new_range + new_min


class GenericFeedForwardNetwork(nn.Module):
    def __init__(self, layers, activations, use_batch_norm=False) -> None:
        super(GenericFeedForwardNetwork, self).__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        self.batch_norm_ops: nn.ModuleList = nn.ModuleList()
        self.activations = activations
        self.use_batch_norm = use_batch_norm

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
            if self.use_batch_norm:
                x = self.batch_norm_ops[i](x)
            activation_func = getattr(F, activation)
            fc_func = self.layers[i]
            x = fc_func(x) if activation == "linear" else activation_func(fc_func(x))
        return x


class DuelingArchitectureQNetwork(nn.Module):
    def __init__(self, layers, activations, use_batch_norm=False, action_dim=0) -> None:
        """
        Dueling Q-Network Architecture: https://arxiv.org/abs/1511.06581

        :param layers: List of layer dimensions
        :param activations: List of layer activations
        :param use_batch_norm: bool indicating whether to apply batch normalization
        :param action_dim: if !=0 use parametric dueling DQN, else standard dueling DQN
        """
        super(DuelingArchitectureQNetwork, self).__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        self.batch_norm_ops: nn.ModuleList = nn.ModuleList()
        self.activations = activations
        self.use_batch_norm = use_batch_norm

        assert len(layers) >= 3, "Invalid layer schema {} for network".format(layers)
        assert (
            len(layers) == len(activations) + 1
        ), "Invalid activation schema {} for network".format(activations)
        assert (
            layers[-2] % 2 == 0
        ), """Last shared layer in dueling architecture should be
        divisible by 2."""

        for i, layer in enumerate(layers[1:-1]):
            self.layers.append(nn.Linear(layers[i], layer))
            self.batch_norm_ops.append(nn.BatchNorm1d(layers[i]))
            guassian_fill_w_gain(self.layers[i].weight, self.activations[i], layers[i])
            init.constant_(self.layers[i].bias, 0)

        # Split last layer into a value & advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(int(layers[-2] + action_dim), int(layers[-2] / 2)),
            nn.ReLU(),
            nn.Linear(int(layers[-2] / 2), layers[-1]),
        )
        self.value = nn.Sequential(
            nn.Linear(int(layers[-2]), int(layers[-2] / 2)),
            nn.ReLU(),
            nn.Linear(int(layers[-2] / 2), 1),
        )

    def forward(self, input) -> torch.FloatTensor:
        if isinstance(input, np.ndarray):
            input = Variable(torch.from_numpy(input))

        state_dim = self.layers[0].in_features
        state = input[:, :state_dim]
        action = input[:, state_dim:]

        x = state
        for i, activation in enumerate(self.activations[:-1]):
            if self.use_batch_norm:
                x = self.batch_norm_ops[i](x)
            activation_func = getattr(F, activation)
            fc_func = self.layers[i]
            x = fc_func(x) if activation == "linear" else activation_func(fc_func(x))

        value = self.value(x)
        x = torch.cat((x, action), dim=1)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean()
