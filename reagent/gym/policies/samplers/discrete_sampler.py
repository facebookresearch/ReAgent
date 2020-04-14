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
    def sample_action(
        self, scores: torch.Tensor, possible_actions_mask: Optional[torch.Tensor] = None
    ) -> rlt.ActorOutput:
        # TODO: temp hack, convert to single instead of batched
        scores = scores.unsqueeze(0)
        assert scores.dim() == 2, (
            "scores dim is %d" % scores.dim()
        )  # batch_size x num_actions
        _, num_actions = scores.shape
        f = F.log_softmax if self.key == "logits" else F.softmax
        if possible_actions_mask is not None:
            assert possible_actions_mask.dim() == 2  # batch_size x num_actions
            mod_scores = f(scores + torch.log(possible_actions_mask))
        else:
            mod_scores = scores
        m = torch.distributions.Categorical(**{self.key: mod_scores / self.temperature})
        raw_action = m.sample()
        action = F.one_hot(raw_action, num_actions).squeeze(0)
        log_prob = m.log_prob(raw_action).float().squeeze(0)
        return rlt.ActorOutput(action=action, log_prob=log_prob)

    @torch.no_grad()
    def log_prob(self, scores: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        m = torch.distributions.Categorical(**{self.key: scores / self.temperature})
        return m.log_prob(action.argmax(dim=-1)).float()


class GreedyActionSampler(Sampler):
    """
    Return the highest scoring action.
    """

    @torch.no_grad()
    def sample_action(
        self, scores: torch.Tensor, possible_actions_mask: Optional[torch.Tensor] = None
    ) -> rlt.ActorOutput:
        # TODO: temp hack
        scores = scores.unsqueeze(0)
        assert scores.dim() == 2, (
            "scores dim is %d" % scores.dim()
        )  # batch_size x num_actions
        _, num_actions = scores.shape
        if possible_actions_mask is not None:
            assert scores.shape == possible_actions_mask.shape
            mod_scores = scores.clone().float()
            mod_scores[~possible_actions_mask.bool()] = -float("inf")
        else:
            mod_scores = scores
        raw_action = mod_scores.argmax(dim=1)
        return rlt.ActorOutput(
            action=F.one_hot(raw_action, num_actions), log_prob=torch.tensor(1.0)
        )

    @torch.no_grad()
    def log_prob(self, scores: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        max_index = scores.argmax(-1)
        match = max_index == action.argmax()
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

    def sample_action(
        self, scores: torch.Tensor, possible_actions_mask: Optional[torch.Tensor] = None
    ) -> rlt.ActorOutput:
        # TODO: temp hack
        scores = scores.unsqueeze(0)
        assert scores.dim() == 2, (
            "scores dim is %d" % scores.dim()
        )  # batch_size x num_actions
        batch_size, num_actions = scores.shape

        if possible_actions_mask is None:
            possible_actions_mask = torch.ones(num_actions)

        argmax = F.one_hot(scores.argmax(dim=1), num_actions).bool()

        p = torch.zeros_like(scores)
        allowed_action_count = float(possible_actions_mask.sum().item())
        mask = torch.repeat_interleave(
            possible_actions_mask.bool().unsqueeze(0), batch_size, axis=0
        )

        rand_prob = self.epsilon / allowed_action_count
        p[mask] = rand_prob

        greedy_prob = 1 - self.epsilon + rand_prob
        p[argmax] = greedy_prob

        m = torch.distributions.Categorical(probs=p)
        raw_action = m.sample()
        action = F.one_hot(raw_action, num_actions).squeeze(0)
        log_prob = m.log_prob(raw_action).squeeze(0)
        return rlt.ActorOutput(action=action, log_prob=log_prob)

    def log_prob(self, scores: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        max_index = self.sample_action(scores).argmax(-1)
        opt = max_index == action.argmax(-1)
        n = len(scores)
        lp = torch.ones(n) * self.epsilon / n
        lp[opt] = 1 - self.epsilon + self.epsilon / n
        return lp

    def update(self) -> None:
        self.epsilon *= self.epsilon_decay
        if self.minimum_epsilon is not None:
            self.epsilon = max(self.epsilon, self.minimum_epsilon)
