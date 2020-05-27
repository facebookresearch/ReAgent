#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.gym.types import Sampler


class SoftmaxActionSampler(Sampler):
    """
    Softmax sampler.
    Equation: http://incompleteideas.net/book/first/ebook/node17.html
    The action scores are logits.

    Args:
        temperature: A measure of how uniformly random the distribution looks.
            The higher the temperature, the more uniform the sampling.
    """

    def __init__(self, temperature: float = 1.0):
        assert temperature > 0, f"Invalid non-positive temperature {temperature}."
        self.temperature = temperature

    def _get_distribution(
        self, scores: torch.Tensor
    ) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=scores / self.temperature)

    @torch.no_grad()
    def sample_action(self, scores: torch.Tensor) -> rlt.ActorOutput:
        assert (
            scores.dim() == 2
        ), f"scores shape is {scores.shape}, not (batch_size, num_actions)"
        batch_size, num_actions = scores.shape
        m = self._get_distribution(scores)
        raw_action = m.sample()
        assert raw_action.shape == (
            batch_size,
        ), f"{raw_action.shape} != ({batch_size}, )"
        action = F.one_hot(raw_action, num_actions)
        assert action.ndim == 2
        log_prob = m.log_prob(raw_action)
        assert log_prob.ndim == 1
        return rlt.ActorOutput(action=action, log_prob=log_prob)

    @torch.no_grad()
    def log_prob(self, scores: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        assert len(scores.shape) == 2, f"{scores.shape}"
        assert scores.shape == action.shape, f"{scores.shape} != {action.shape}"
        m = self._get_distribution(scores)
        # pyre-fixme[16]: `Tensor` has no attribute `argmax`.
        return m.log_prob(action.argmax(dim=1))


class GreedyActionSampler(Sampler):
    """
    Return the highest scoring action.
    """

    def _get_greedy_indices(self, scores: torch.Tensor) -> torch.Tensor:
        assert (
            len(scores.shape) == 2
        ), f"scores shape is {scores.shape}, not (batchsize, num_actions)"
        # pyre-fixme[16]: `Tensor` has no attribute `argmax`.
        return scores.argmax(dim=1)

    @torch.no_grad()
    def sample_action(self, scores: torch.Tensor) -> rlt.ActorOutput:

        batch_size, num_actions = scores.shape
        raw_action = self._get_greedy_indices(scores)
        action = F.one_hot(raw_action, num_actions)
        assert action.shape == (batch_size, num_actions)
        return rlt.ActorOutput(action=action, log_prob=torch.ones_like(raw_action))

    @torch.no_grad()
    def log_prob(self, scores: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        greedy_indices = self._get_greedy_indices(scores)
        # pyre-fixme[16]: `Tensor` has no attribute `argmax`.
        match = greedy_indices == action.argmax(-1)
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
        lp[opt] = 1 - self.epsilon + self.epsilon / n
        return lp

    def update(self) -> None:
        self.epsilon *= self.epsilon_decay
        if self.minimum_epsilon is not None:
            self.epsilon = max(self.epsilon, self.minimum_epsilon)
