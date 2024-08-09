#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from reagent.models.base import ModelBase
from reagent.models.fully_connected_network import (
    ACTIVATION_MAP,
    gaussian_fill_w_gain,
    SlateBatchNorm1d,
)
from torch.distributions import Normal


logger = logging.getLogger(__name__)

# code based off of online tutorial at: https://github.com/cpark321/uncertainty-deep-learning/blob/master/01.%20Bayes-by-Backprop.ipynb


class LinearBBB(ModelBase):
    """
    Layer of our BNN.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        activation,
        orthogonal_init: bool = False,
        min_std: float = 0.0,
        prior_var: float = 1.0,
    ) -> None:
        """
        Initialization of our layer : our prior is a normal distribution
        centered in 0 and of variance 20.
        """
        # initialize layers
        super().__init__()
        # set input and output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(torch.zeros(output_dim, input_dim))
        self.w_rho = nn.Parameter(torch.zeros(output_dim, input_dim))

        gain = torch.nn.init.calculate_gain(activation)
        if orthogonal_init:
            # provably better https://openreview.net/forum?id=rkgqN1SYvr
            nn.init.orthogonal_(self.w_mu.data, gain=gain)
            nn.init.orthogonal_(self.w_rho.data, gain=gain)
        else:
            # gaussian init
            gaussian_fill_w_gain(
                self.w_mu, gain=gain, dim_in=input_dim, min_std=min_std
            )
            gaussian_fill_w_gain(
                self.w_rho, gain=gain, dim_in=input_dim, min_std=min_std
            )

        # initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(torch.zeros(output_dim))
        self.b_rho = nn.Parameter(torch.zeros(output_dim))

        # initialize weight samples (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None
        self.log_prior = None
        self.w_post = None
        self.b_post = None
        self.log_post = None

        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0, prior_var)

    def forward(self, x: torch.Tensor):
        """
        Optimization process
        """
        # sample weights
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon

        # sample bias
        b_epsilon = Normal(0, 1).sample(self.b_mu.shape)
        self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon

        # record log prior by evaluating log pdf of prior at sampled weight and bias
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # record log variational posterior by evaluating log pdf of normal distribution defined by parameters with respect at the sampled values
        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1 + torch.exp(self.b_rho)))
        self.log_post = (
            self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()
        )

        return F.linear(x, self.w, self.b)


class FullyConnectedProbabilisticNetwork(ModelBase):
    def __init__(
        self,
        layers,
        activations,
        prior_var,
        *,
        noise_tol: float = 0.1,
        use_batch_norm: bool = False,
        min_std: float = 0.0,
        dropout_ratio: float = 0.0,
        use_layer_norm: bool = False,
        normalize_output: bool = False,
        orthogonal_init: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = layers[0]
        self.use_batch_norm = use_batch_norm

        modules: List[nn.Module] = []
        linear_bbbs: List[nn.Module] = []
        self.noise_tol = noise_tol
        self.layers = layers

        assert len(layers) == len(activations) + 1

        for i, ((in_dim, out_dim), activation) in enumerate(
            zip(zip(layers, layers[1:]), activations)
        ):
            # Add BatchNorm1d
            if use_batch_norm:
                modules.append(SlateBatchNorm1d(in_dim))
            # Add Linear
            linear_bbb = LinearBBB(in_dim, out_dim, activation, prior_var=prior_var)
            linear_bbbs.append(linear_bbb)
            # assuming activation is valid

            modules.append(linear_bbb)
            # Add LayerNorm
            if use_layer_norm and (normalize_output or i < len(activations) - 1):
                modules.append(nn.LayerNorm(out_dim))  # type: ignore
            # Add activation
            if activation in ACTIVATION_MAP:
                modules.append(ACTIVATION_MAP[activation]())
            else:
                # See if it matches any of the nn modules
                modules.append(getattr(nn, activation)())
            # Add Dropout
            if dropout_ratio > 0.0 and (normalize_output or i < len(activations) - 1):
                modules.append(nn.Dropout(p=dropout_ratio))

        self.dnn = nn.Sequential(*modules)  # type: ignore
        self.linear_bbbs = nn.ModuleList(linear_bbbs)

    def input_prototype(self):
        return torch.randn(1, self.input_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass for generic feed-forward DNNs. Assumes activation names
        are valid pytorch activation names.
        :param input tensor
        """
        return self.dnn(input)

    def log_prior(self):
        # calculate the log prior over all the layers
        ret = 0
        for x in self.linear_bbbs:
            ret += x.log_prior
        return ret

    def log_post(self) -> torch.Tensor:
        # calculate the log posterior over all the layers
        ret = 0
        for x in self.linear_bbbs:
            ret += x.log_post
        # pyre-fixme[7]: Expected `Tensor` but got `int`.
        return ret

    def sample_elbo(self, input: torch.Tensor, target: torch.Tensor, num_samples: int):
        # rlt.BanditRewardModelInput

        # we calculate the negative elbo, which will be our loss function
        # initialize tensors
        outputs = torch.zeros(num_samples, target.shape[0])
        log_priors = torch.zeros(num_samples)
        log_posts = torch.zeros(num_samples)
        log_likes = torch.zeros(num_samples)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(num_samples):
            outputs[i] = self(input).reshape(-1)  # make predictions
            log_priors[i] = self.log_prior()  # get log prior
            log_posts[i] = self.log_post()  # get log variational posterior
            log_likes[i] = (
                Normal(outputs[i], self.noise_tol).log_prob(target.reshape(-1)).sum()
            )  # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        loss = log_post - log_prior - log_like
        return loss
