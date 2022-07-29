#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math
from typing import List, Optional

import torch
from reagent.core import types as rlt
from reagent.core.parameters import CONTINUOUS_TRAINING_ACTION_RANGE
from reagent.core.tensorboardX import SummaryWriterContext
from reagent.models.base import ModelBase
from reagent.models.fully_connected_network import FullyConnectedNetwork
from torch.distributions import Dirichlet
from torch.distributions.normal import Normal

LOG_PROB_MIN: float = -2.0
LOG_PROB_MAX = 2.0


class StochasticActor(ModelBase):
    """
    An actor defined by a scorer and sampler.
    The scorer gives each action an score. And the sampler samples actions
    based on the action scores."""

    def __init__(self, scorer, sampler) -> None:
        super().__init__()
        self.scorer = scorer
        self.sampler = sampler

    def input_prototype(self):
        return self.scorer.input_prototype()

    def get_distributed_data_parallel_model(self):
        raise NotImplementedError()

    def forward(self, state):
        action_scores = self.scorer(state)
        return self.sampler.sample_action(action_scores, possible_actions_mask=None)


class FullyConnectedActor(ModelBase):
    """
    A general model arch for mapping from state to ActorOutput,
    which contains an action tensor and log probabilities (optional).

    The model arch is often used to implement the actor in actor-critic algorithms.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sizes: List[int],
        activations: List[str],
        use_batch_norm: bool = False,
        action_activation: str = "tanh",
        exploration_variance: Optional[float] = None,
    ) -> None:
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

        # Gaussian noise for exploration.
        self.exploration_variance = exploration_variance
        if exploration_variance is not None:
            assert exploration_variance > 0
            loc = torch.zeros(action_dim).float()
            scale = torch.ones(action_dim).float() * exploration_variance
            self.noise_dist = Normal(loc=loc, scale=scale)

    def input_prototype(self):
        return rlt.FeatureData(torch.randn(1, self.state_dim))

    def forward(self, state: rlt.FeatureData) -> rlt.ActorOutput:
        action = self.fc(state.float_features)
        batch_size = action.shape[0]
        assert action.shape == (
            batch_size,
            self.action_dim,
        ), f"{action.shape} != ({batch_size}, {self.action_dim})"

        if self.exploration_variance is None:
            log_prob = torch.zeros(batch_size).to(action.device).float().view(-1, 1)
            return rlt.ActorOutput(action=action, log_prob=log_prob)

        noise = self.noise_dist.sample((batch_size,))
        # TODO: log prob is affected by clamping, how to handle that?
        log_prob = (
            self.noise_dist.log_prob(noise).to(action.device).sum(dim=1).view(-1, 1)
        ).clamp(LOG_PROB_MIN, LOG_PROB_MAX)
        action = (action + noise.to(action.device)).clamp(
            *CONTINUOUS_TRAINING_ACTION_RANGE
        )
        return rlt.ActorOutput(action=action, log_prob=log_prob)


class GaussianFullyConnectedActor(ModelBase):
    """
    A model arch similar to FullyConnectedActor, except that the last layer
    represents the parameters of a multi-dimensional Gaussian distribution.
    The action to return is sampled from the Gaussian distribution.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sizes: List[int],
        activations: List[str],
        scale: float = 0.05,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        use_l2_normalization: bool = False,
    ) -> None:
        """
        Args:
            use_l2_normalization: if True, divides action by l2 norm.
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
        # The last layer is mean & scale for reparameterization trick
        self.fc = FullyConnectedNetwork(
            [state_dim] + sizes + [action_dim * 2],
            activations + ["linear"],
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.loc_layer_norm = torch.nn.LayerNorm(action_dim)
            self.scale_layer_norm = torch.nn.LayerNorm(action_dim)

        self.use_l2_normalization = use_l2_normalization

        # used to calculate log-prob
        self.const = math.log(math.sqrt(2 * math.pi))
        self.eps = 1e-6

    def input_prototype(self):
        return rlt.FeatureData(torch.randn(1, self.state_dim))

    def _normal_log_prob(self, r, scale_log):
        """
        Compute log probability from normal distribution the same way as
        torch.distributions.normal.Normal, which is:

        ```
        -((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
        ```

        In the context of this class, `value = loc + r * scale`. Therefore, this
        function only takes `r` & `scale`; it can be reduced to below.

        The primary reason we don't use Normal class is that it currently
        cannot be exported through ONNX.
        """
        return -(r**2) / 2 - scale_log - self.const

    def _squash_correction(self, squashed_action):
        """
        Same as
        https://github.com/haarnoja/sac/blob/108a4229be6f040360fcca983113df9c4ac23a6a/sac/policies/gaussian_policy.py#L133
        """
        return (1 - squashed_action**2 + self.eps).log()

    def _get_loc_and_scale_log(self, state: rlt.FeatureData):
        loc_scale = self.fc(state.float_features)
        loc = loc_scale[::, : self.action_dim]
        scale_log = loc_scale[::, self.action_dim :]

        if self.use_layer_norm:
            loc = self.loc_layer_norm(loc)
            scale_log = self.scale_layer_norm(scale_log)

        scale_log = scale_log.clamp(LOG_PROB_MIN, LOG_PROB_MAX)
        return loc, scale_log

    def _squash_raw_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        # NOTE: without clamping to (-(1-eps), 1-eps), torch.tanh would output
        # 1, and torch.atanh would map it to +inf, causing log_prob to be -inf.
        squashed_action = torch.clamp(
            torch.tanh(raw_action), -1.0 + self.eps, 1.0 - self.eps
        )
        if self.use_l2_normalization:
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            l2_norm = (squashed_action**2).sum(dim=1, keepdim=True).sqrt()
            squashed_action = squashed_action / l2_norm
        return squashed_action

    def forward(self, state: rlt.FeatureData):
        loc, scale_log = self._get_loc_and_scale_log(state)
        r = torch.randn_like(scale_log, device=scale_log.device)
        raw_action = loc + r * scale_log.exp()
        squashed_action = self._squash_raw_action(raw_action)
        squashed_loc = self._squash_raw_action(loc)
        if SummaryWriterContext._global_step % 1000 == 0:
            SummaryWriterContext.add_histogram("actor/forward/loc", loc.detach().cpu())
            SummaryWriterContext.add_histogram(
                "actor/forward/scale_log", scale_log.detach().cpu()
            )

        return rlt.ActorOutput(
            action=squashed_action,
            log_prob=self.get_log_prob(state, squashed_action),
            squashed_mean=squashed_loc,
        )

    def get_log_prob(self, state: rlt.FeatureData, squashed_action: torch.Tensor):
        """
        Action is expected to be squashed with tanh
        """
        if self.use_l2_normalization:
            # TODO: calculate log_prob for l2 normalization
            # https://math.stackexchange.com/questions/3120506/on-the-distribution-of-a-normalized-gaussian-vector
            # http://proceedings.mlr.press/v100/mazoure20a/mazoure20a.pdf
            pass

        loc, scale_log = self._get_loc_and_scale_log(state)
        raw_action = torch.atanh(squashed_action)
        r = (raw_action - loc) / scale_log.exp()
        log_prob = self._normal_log_prob(r, scale_log)
        squash_correction = self._squash_correction(squashed_action)
        if SummaryWriterContext._global_step % 1000 == 0:
            SummaryWriterContext.add_histogram(
                "actor/get_log_prob/loc", loc.detach().cpu()
            )
            SummaryWriterContext.add_histogram(
                "actor/get_log_prob/scale_log", scale_log.detach().cpu()
            )
            SummaryWriterContext.add_histogram(
                "actor/get_log_prob/log_prob", log_prob.detach().cpu()
            )
            SummaryWriterContext.add_histogram(
                "actor/get_log_prob/squash_correction", squash_correction.detach().cpu()
            )
        return torch.sum(log_prob - squash_correction, dim=1).reshape(-1, 1)


class DirichletFullyConnectedActor(ModelBase):
    """
    A model arch similar to FullyConnectedActor, except that the last layer
    is sampled from a Dirichlet distribution
    """

    # Used to prevent concentration from being 0
    EPSILON = 1e-6

    def __init__(
        self, state_dim, action_dim, sizes, activations, use_batch_norm: bool = False
    ) -> None:
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
            activations + ["linear"],
            use_batch_norm=use_batch_norm,
        )

    def input_prototype(self):
        return rlt.FeatureData(torch.randn(1, self.state_dim))

    def _get_concentration(self, state):
        """
        Get concentration of distribution.
        https://stats.stackexchange.com/questions/244917/what-exactly-is-the-alpha-in-the-dirichlet-distribution
        """
        return self.fc(state.float_features).exp() + self.EPSILON

    @torch.no_grad()
    def get_log_prob(self, state, action):
        concentration = self._get_concentration(state)
        log_prob = Dirichlet(concentration).log_prob(action)
        return log_prob.unsqueeze(dim=1)

    def forward(self, state):
        concentration = self._get_concentration(state)
        if self.training:
            # PyTorch can't backwards pass _sample_dirichlet
            action = Dirichlet(concentration).rsample()
        else:
            # ONNX can't export Dirichlet()
            action = torch._sample_dirichlet(concentration)

        log_prob = Dirichlet(concentration).log_prob(action)
        return rlt.ActorOutput(action=action, log_prob=log_prob.unsqueeze(dim=1))
