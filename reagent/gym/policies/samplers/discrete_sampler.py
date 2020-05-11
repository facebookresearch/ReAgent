#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional

import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.gym.types import Sampler


class SoftmaxActionSampler(Sampler):
    """Softmax sampler - sample actions proportional to action probability.

    The unnormalized action probabilities can be expressed directly or as logits.
    Use the default logits if the scores can be negative.

    Args:
        temperature: Float [default: 1.0] The higher the temperature,
         the more the sampling looks uniform
        probs: bool [default: False] If selected, scores are interpreted
         to be (unnormalized) probabilities instead of logits
    """

    def __init__(self, temperature: float = 1.0, probs: bool = False):
        self.temperature = temperature
        self.key = "probs" if probs else "logits"

    @torch.no_grad()
    def sample_action(self, scores: torch.Tensor) -> rlt.ActorOutput:
        assert scores.dim() == 2, (
            "scores dim is %d" % scores.dim()
        )  # batch_size x num_actions
        _, num_actions = scores.shape
        f = F.log_softmax if self.key == "logits" else F.softmax
        m = torch.distributions.Categorical(**{self.key: scores / self.temperature})
        raw_action = m.sample()
        assert raw_action.ndim == 1
        action = F.one_hot(raw_action, num_actions)
        assert action.ndim == 2
        log_prob = m.log_prob(raw_action).float()
        assert log_prob.ndim == 1
        return rlt.ActorOutput(action=action, log_prob=log_prob)

    @torch.no_grad()
    def log_prob(self, scores: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # pyre-fixme[16]: `Tensor` has no attribute `ndim`.
        assert scores.ndim == 2
        m = torch.distributions.Categorical(**{self.key: scores / self.temperature})
        # pyre-fixme[16]: `Tensor` has no attribute `argmax`.
        return m.log_prob(action.argmax(dim=-1)).float()


class GreedyActionSampler(Sampler):
    """
    Return the highest scoring action.
    """

    @torch.no_grad()
    def sample_action(self, scores: torch.Tensor) -> rlt.ActorOutput:
        assert scores.dim() == 2, (
            "scores dim is %d" % scores.dim()
        )  # batch_size x num_actions
        batch_size, num_actions = scores.shape
        # pyre-fixme[16]: `Tensor` has no attribute `argmax`.
        raw_action = scores.argmax(dim=1)
        assert raw_action.ndim == 1
        action = F.one_hot(raw_action, num_actions)
        assert action.shape == (batch_size, num_actions)
        log_prob = torch.ones(batch_size, device=scores.device)
        return rlt.ActorOutput(action=action, log_prob=log_prob)

    @torch.no_grad()
    def log_prob(self, scores: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # pyre-fixme[16]: `Tensor` has no attribute `argmax`.
        max_index = scores.argmax(-1)
        match = max_index == action.argmax(-1)
        lp = torch.zeros(scores.shape[0]).float()
        lp[match] = -float("inf")
        return lp


class EpsilonGreedyActionSampler(Sampler):
    """
    Epsilon-Greedy Policy

    With probability epsilon, a random action is sampled. Otherwise,
    the highest scoring (greedy) action is chosen.

    Call update() to decay the amount of exploration by lowering
    `epsilon` by a factor of `epsilon_decay` (<=1) until we reach
    `minimum_epsilon`
    """

    def __init__(
        self, epsilon: float, epsilon_decay: float = 1.0, minimum_epsilon: float = 0.0
    ):
        self.epsilon = float(epsilon)
        assert epsilon_decay <= 1
        self.epsilon_decay = epsilon_decay
        assert minimum_epsilon <= epsilon_decay
        self.minimum_epsilon = minimum_epsilon

    def sample_action(self, scores: torch.Tensor) -> rlt.ActorOutput:
        assert scores.dim() == 2, (
            "scores dim is %d" % scores.dim()
        )  # batch_size x num_actions
        batch_size, num_actions = scores.shape

        # pyre-fixme[16]: `Tensor` has no attribute `argmax`.
        argmax = F.one_hot(scores.argmax(dim=1), num_actions).bool()

        rand_prob = self.epsilon / num_actions
        p = torch.full_like(rand_prob, scores)

        greedy_prob = 1 - self.epsilon + rand_prob
        p[argmax] = greedy_prob

        m = torch.distributions.Categorical(probs=p)
        raw_action = m.sample()
        action = F.one_hot(raw_action, num_actions)
        assert action.shape == (batch_size, num_actions)
        log_prob = m.log_prob(raw_action)
        assert log_prob.shape == (batch_size,)
        return rlt.ActorOutput(action=action, log_prob=log_prob)

    def log_prob(self, scores: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        max_index = self.sample_action(scores).argmax(-1)
        # pyre-fixme[16]: `Tensor` has no attribute `argmax`.
        opt = max_index == action.argmax(-1)
        n = len(scores)
        lp = torch.ones(n) * self.epsilon / n
        # pyre-fixme[16]: `float` has no attribute `__setitem__`.
        lp[opt] = 1 - self.epsilon + self.epsilon / n
        # pyre-fixme[7]: Expected `Tensor` but got `float`.
        return lp

    def update(self) -> None:
        self.epsilon *= self.epsilon_decay
        if self.minimum_epsilon is not None:
            self.epsilon = max(self.epsilon, self.minimum_epsilon)
