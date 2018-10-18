#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


logger = logging.getLogger(__name__)


class NoisyLinear(nn.Linear):
    def __init__(self, in_dimension, out_dimension, std_dev_init=0.4) -> None:
        """
        Noisy Networks for Exploration: https://arxiv.org/abs/1706.10295
        Standard linear layer: y = wx + b
        Noise linear layer: y = wx + b, where:
            w = u1 + σ1 * ε and b = u2 + σ2 * ε
            ε ~ N(0, std)
        Using factorized Guassian Noise for performance.

        :param in_dimension: Number of in dimensions
        :param out_dimension: Number of out dimensions
        :param std_dev_init: Standard deviation initialization value
        """
        self.in_dimension, self.out_dimension = in_dimension, out_dimension
        super(NoisyLinear, self).__init__(
            self.in_dimension, self.out_dimension, bias=True
        )
        sigma_init = std_dev_init / math.sqrt(in_dimension)
        self.sigma_weight = nn.Parameter(
            torch.Tensor(self.out_dimension, self.in_dimension).fill_(sigma_init)
        )
        self.sigma_bias = nn.Parameter(
            torch.Tensor(self.out_dimension).fill_(sigma_init)
        )

    def forward(self, input):
        # Hack: Force noise vectors to be function of input so they are put into
        # predict_net and not init_net when tracing with ONNX
        epsilon_input = torch.randn(1, input.size()[1], device=input.device)
        epsilon_output = torch.randn(
            self.out_dimension - input.size()[1] + input.size()[1],
            1,
            device=input.device,
        )
        epsilon_in = torch.sign(epsilon_input) * torch.sqrt(torch.abs(epsilon_input))
        epsilon_out = torch.sign(epsilon_output) * torch.sqrt(torch.abs(epsilon_output))

        # Add noise to bias and weights
        noise = torch.mul(epsilon_in, epsilon_out)
        bias = self.bias + self.sigma_bias * epsilon_out.t()
        weight = self.weight + self.sigma_weight * noise
        return input.matmul(weight.t()) + bias
