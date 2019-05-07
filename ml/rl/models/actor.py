#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math

import torch
from ml.rl import types as rlt
from ml.rl.models.base import ModelBase
from ml.rl.models.fully_connected_network import FullyConnectedNetwork
from torch.distributions import Dirichlet
from torch.distributions.normal import Normal


class ActorWithPreprocessing(ModelBase):
    def __init__(self, actor_network, state_preprocessor):
        super().__init__()
        self.state_preprocessor = state_preprocessor
        self.actor_network = actor_network

    def forward(self, input):
        preprocessed_state = self.state_preprocessor(input.state)
        return self.actor_network(rlt.StateInput(state=preprocessed_state))

    def input_prototype(self):
        return rlt.StateInput(state=self.state_preprocessor.input_prototype())


class FullyConnectedActor(ModelBase):
    def __init__(
        self,
        state_dim,
        action_dim,
        sizes,
        activations,
        use_batch_norm=False,
        action_activation="tanh",
    ):
        super().__init__()
        assert state_dim > 0, "state_dim must be > 0, got {}".format(state_dim)
        assert action_dim > 0, "action_dim must be > 0, got {}".format(action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        assert len(sizes) == len(
            activations
        ), "The numbers of sizes and activations must match; got {} vs {}".format(
            len(sizes), len(activations)
        )
        self.action_activation = action_activation
        self.fc = FullyConnectedNetwork(
            [state_dim] + sizes + [action_dim],
            activations + [self.action_activation],
            use_batch_norm=use_batch_norm,
        )

    def input_prototype(self):
        return rlt.StateInput(
            state=rlt.FeatureVector(float_features=torch.randn(1, self.state_dim))
        )

    def forward(self, input):
        action = self.fc(input.state.float_features)
        return rlt.ActorOutput(action=action)


class GaussianFullyConnectedActor(ModelBase):
    def __init__(
        self,
        state_dim,
        action_dim,
        sizes,
        activations,
        scale=0.05,
        use_batch_norm=False,
    ):
        super().__init__()
        assert state_dim > 0, "state_dim must be > 0, got {}".format(state_dim)
        assert action_dim > 0, "action_dim must be > 0, got {}".format(action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        assert len(sizes) == len(
            activations
        ), "The numbers of sizes and activations must match; got {} vs {}".format(
            len(sizes), len(activations)
        )
        # The last layer is mean & scale for reparamerization trick
        self.fc = FullyConnectedNetwork(
            [state_dim] + sizes + [action_dim * 2],
            activations + ["linear"],
            use_batch_norm=use_batch_norm,
        )

        # used to calculate log-prob
        self.const = math.log(math.sqrt(2 * math.pi))
        self.eps = 1e-6
        self._log_min_max = (-20.0, 2.0)

    def input_prototype(self):
        return rlt.StateInput(
            state=rlt.FeatureVector(float_features=torch.randn(1, self.state_dim))
        )

    def _log_prob(self, r, scale_log):
        """
        Compute log probability from normal distribution the same way as
        torch.distributions.normal.Normal, which is:

        ```
        -((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
        ```

        In the context of this class, `value = loc + r * scale`. Therefore, this function
        only takes `r` & `scale`; it can be reduced to below.

        The primary reason we don't use Normal class is that it currently
        cannot be exported through ONNX.
        """
        return -(r ** 2) / 2 - scale_log - self.const

    def _squash_correction(self, squashed_action):
        """
        Same as
        https://github.com/haarnoja/sac/blob/108a4229be6f040360fcca983113df9c4ac23a6a/sac/policies/gaussian_policy.py#L133
        """
        return (1 - squashed_action ** 2 + self.eps).log()

    def _get_loc_and_scale_log(self, state):
        loc_scale = self.fc(state.float_features)
        loc = loc_scale[::, : self.action_dim]
        scale_log = loc_scale[::, self.action_dim :].clamp(*self._log_min_max)
        return loc, scale_log

    def forward(self, input):
        loc, scale_log = self._get_loc_and_scale_log(input.state)
        r = torch.randn_like(scale_log, device=scale_log.device)
        action = torch.tanh(loc + r * scale_log.exp())
        if not self.training:
            # ONNX doesn't like reshape either..
            return rlt.ActorOutput(action=action)
        # Since each dim are independent, log-prob is simply sum
        log_prob = torch.sum(
            self._log_prob(r, scale_log) - self._squash_correction(action), dim=1
        )
        return rlt.ActorOutput(action=action, log_prob=log_prob.reshape(-1, 1))

    def _atanh(self, x):
        """
        Can't find this on pytorch doc :(
        """
        return ((1 + x).log() - (1 - x).log()) / 2

    def get_log_prob(self, state, squashed_action):
        """
        Action is expected to be squashed with tanh
        """
        with torch.no_grad():
            loc, scale_log = self._get_loc_and_scale_log(state)
            # This is not getting exported; we can use it
            n = Normal(loc, scale_log.exp())
            raw_action = self._atanh(squashed_action)
            log_prob = torch.sum(
                n.log_prob(raw_action) - self._squash_correction(squashed_action), dim=1
            ).reshape(-1, 1)

        return log_prob


class DirichletFullyConnectedActor(ModelBase):
    def __init__(self, state_dim, action_dim, sizes, activations, use_batch_norm=False):
        """
        AKA the multivariate beta distribution. Used in cases where actor's action
        must sum to 1.
        """
        super().__init__()
        assert state_dim > 0, "state_dim must be > 0, got {}".format(state_dim)
        assert action_dim > 0, "action_dim must be > 0, got {}".format(action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        assert len(sizes) == len(
            activations
        ), "The numbers of sizes and activations must match; got {} vs {}".format(
            len(sizes), len(activations)
        )

        # The last layer gives the concentration of the distribution.
        self.fc = FullyConnectedNetwork(
            [state_dim] + sizes + [action_dim],
            activations + ["relu"],
            use_batch_norm=use_batch_norm,
        )

    def input_prototype(self):
        return rlt.StateInput(
            state=rlt.FeatureVector(float_features=torch.randn(1, self.state_dim))
        )

    def _get_concentration(self, state):
        """
        Get concentration of distribution.
        https://stats.stackexchange.com/questions/244917/what-exactly-is-the-alpha-in-the-dirichlet-distribution
        """
        return self.fc(state.float_features)

    def get_log_prob(self, state, action):
        with torch.no_grad():
            concentration = self._get_concentration(state)
            # This is not getting exported; we can use it
            dist = Dirichlet(concentration)
            log_prob = dist.log_prob(action)
        return log_prob

    def forward(self, input):
        concentration = self._get_concentration(input.state)
        action = torch._sample_dirichlet(concentration)

        if not self.training:
            # ONNX doesn't like reshape either..
            return rlt.ActorOutput(action=action)

        log_prob = self.get_log_prob(input.state, action)
        return rlt.ActorOutput(action=action, log_prob=log_prob.unsqueeze(dim=1))
