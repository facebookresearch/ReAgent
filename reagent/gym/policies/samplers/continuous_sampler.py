#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import reagent.types as rlt
import torch
from reagent.gym.types import GaussianSamplerScore, Sampler


class GaussianSampler(Sampler):
    def __init__(self, actor_network):
        self.actor_network = actor_network

    def _sample_action(self, loc: torch.Tensor, scale_log: torch.Tensor):
        r = torch.randn_like(scale_log, device=scale_log.device)
        # pyre-fixme[16]: `Tensor` has no attribute `exp`.
        action = torch.tanh(loc + r * scale_log.exp())
        # Since each dim are independent, log-prob is simply sum
        log_prob = self.actor_network._log_prob(r, scale_log)
        squash_correction = self.actor_network._squash_correction(action)
        log_prob = torch.sum(log_prob - squash_correction, dim=1)
        return action, log_prob

    @torch.no_grad()
    def sample_action(self, scores: GaussianSamplerScore) -> rlt.ActorOutput:
        self.actor_network.eval()
        unscaled_actions, log_prob = self._sample_action(scores.loc, scores.scale_log)
        self.actor_network.train()

        return rlt.ActorOutput(action=unscaled_actions, log_prob=log_prob)

    def _log_prob(
        self, loc: torch.Tensor, scale_log: torch.Tensor, squashed_action: torch.Tensor
    ):
        # This is not getting exported; we can use it
        # pyre-fixme[16]: `Tensor` has no attribute `exp`.
        n = torch.distributions.Normal(loc, scale_log.exp())
        raw_action = self.actor_network._atanh(squashed_action)
        log_prob = n.log_prob(raw_action)
        squash_correction = self.actor_network._squash_correction(squashed_action)
        log_prob = torch.sum(log_prob - squash_correction, dim=1)
        return log_prob

    @torch.no_grad()
    # pyre-fixme[14]: `log_prob` overrides method defined in `Sampler` inconsistently.
    def log_prob(
        self, scores: GaussianSamplerScore, squashed_action: torch.Tensor
    ) -> torch.Tensor:
        self.actor_network.eval()
        # pyre-fixme[20]: Argument `squashed_action` expected.
        log_prob = self._log_prob(scores.loc, scores.scale_log)
        self.actor_network.train()
        return log_prob
