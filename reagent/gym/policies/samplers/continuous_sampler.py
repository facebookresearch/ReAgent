#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import reagent.types as rlt
import torch
from reagent.gym.types import GaussianSamplerScore, Sampler
from reagent.tensorboardX import SummaryWriterContext
from reagent.torch_utils import rescale_torch_tensor


class GaussianSampler(Sampler):
    def __init__(
        self,
        actor_network,
        min_serving_action,
        max_serving_action,
        min_training_action,
        max_training_action,
    ):
        self.actor_network = actor_network
        self.min_serving_action = min_serving_action
        self.max_serving_action = max_serving_action
        self.min_training_action = min_training_action
        self.max_training_action = max_training_action

    def _sample_action(self, loc: torch.Tensor, scale_log: torch.Tensor):
        r = torch.randn_like(scale_log, device=scale_log.device)
        action = torch.tanh(loc + r * scale_log.exp())
        # Since each dim are independent, log-prob is simply sum
        log_prob = self.actor_network._log_prob(r, scale_log)
        squash_correction = self.actor_network._squash_correction(action)
        if SummaryWriterContext._global_step % 1000 == 0:
            SummaryWriterContext.add_histogram("actor/forward/loc", loc.detach().cpu())
            SummaryWriterContext.add_histogram(
                "actor/forward/scale_log", scale_log.detach().cpu()
            )
            SummaryWriterContext.add_histogram(
                "actor/forward/log_prob", log_prob.detach().cpu()
            )
            SummaryWriterContext.add_histogram(
                "actor/forward/squash_correction", squash_correction.detach().cpu()
            )
        log_prob = torch.sum(log_prob - squash_correction, dim=1)
        return action, log_prob

    @torch.no_grad()
    def sample_action(self, scores: GaussianSamplerScore) -> rlt.ActorOutput:
        self.actor_network.eval()
        action, log_prob = self._sample_action(scores.loc, scores.scale_log)

        # clamp actions to make sure actions are in the range
        clamped_actions = torch.max(
            torch.min(action, self.max_training_action), self.min_training_action
        )
        rescaled_actions = rescale_torch_tensor(
            clamped_actions,
            new_min=self.min_serving_action,
            new_max=self.max_serving_action,
            prev_min=self.min_training_action,
            prev_max=self.max_training_action,
        )
        self.actor_network.train()
        return rlt.ActorOutput(action=rescaled_actions, log_prob=log_prob)

    def _log_prob(
        self, loc: torch.Tensor, scale_log: torch.Tensor, squashed_action: torch.Tensor
    ):
        # This is not getting exported; we can use it
        n = torch.distributions.Normal(loc, scale_log.exp())
        raw_action = self.actor_network._atanh(squashed_action)
        log_prob = n.log_prob(raw_action)
        squash_correction = self.actor_network._squash_correction(squashed_action)
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
        log_prob = torch.sum(log_prob - squash_correction, dim=1)
        return log_prob

    @torch.no_grad()
    def log_prob(
        self, scores: GaussianSamplerScore, squashed_action: torch.Tensor
    ) -> torch.Tensor:
        self.actor_network.eval()
        log_prob = self._log_prob(scores.loc, scores.scale_log)
        self.actor_network.train()
        return log_prob
