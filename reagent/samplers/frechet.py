#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import Optional

import reagent.types as rlt
import torch
from reagent.core.configuration import resolve_defaults
from reagent.gym.types import Sampler
from torch.distributions import Gumbel


class FrechetSort(Sampler):
    @resolve_defaults
    def __init__(
        self,
        shape: float = 1.0,
        topk: Optional[int] = None,
        equiv_len: Optional[int] = None,
        log_scores: bool = False,
    ):
        """FréchetSort is a softer version of descending sort which samples all possible
        orderings of items favoring orderings which resemble descending sort. This can
        be used to convert descending sort by rank score into a differentiable,
        stochastic policy amenable to policy gradient algorithms.

        :param shape: parameter of Frechet Distribution. Lower values correspond to
        aggressive deviations from descending sort.
        :param topk: If specified, only the first topk actions are specified.
        :param equiv_len: Orders are considered equivalent if the top equiv_len match. Used
            in probability computations
        :param log_scores Scores passed in are already log-transformed. In this case, we would
        simply add Gumbel noise.

        Example:

        Consider the sampler:

        sampler = FrechetSort(shape=3, topk=5, equiv_len=3)

        Given a set of scores, this sampler will produce indices of items roughly
        resembling a argsort by scores in descending order. The higher the shape,
        the more it would resemble a descending argsort. `topk=5` means only the top
        5 ranks will be output. The `equiv_len` determines what orders are considered
        equivalent for probability computation. In this example, the sampler will
        produce probability for the top 3 items appearing in a given order for the
        `log_prob` call.
        """
        self.shape = shape
        self.topk = topk
        self.upto = equiv_len
        if topk is not None:
            if equiv_len is None:
                self.upto = topk
            # pyre-fixme[58]: `>` is not supported for operand types `Optional[int]`
            #  and `Optional[int]`.
            if self.upto > self.topk:
                raise ValueError(f"Equiv length {equiv_len} cannot exceed topk={topk}.")
        self.gumbel_noise = Gumbel(0, 1.0 / shape)
        self.log_scores = log_scores

    @staticmethod
    def select_indices(scores: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Helper for scores[actions] that are also works for batched tensors"""
        if len(actions.shape) > 1:
            num_rows = scores.size(0)
            row_indices = torch.arange(num_rows).unsqueeze(0).T  # pyre-ignore[ 16 ]
            return scores[row_indices, actions].T
        else:
            return scores[actions]

    def sample_action(self, scores: torch.Tensor) -> rlt.ActorOutput:
        """Sample a ranking according to Frechet sort. Note that possible_actions_mask
        is ignored as the list of rankings scales exponentially with slate size and
        number of items and it can be difficult to enumerate them."""
        assert scores.dim() == 2, "sample_action only accepts batches"
        log_scores = scores if self.log_scores else torch.log(scores)
        perturbed = log_scores + self.gumbel_noise.sample(scores.shape)
        action = torch.argsort(perturbed.detach(), descending=True)
        if self.topk is not None:
            action = action[: self.topk]
        log_prob = self.log_prob(scores, action)
        return rlt.ActorOutput(action, log_prob)

    def log_prob(self, scores: torch.Tensor, action) -> torch.Tensor:
        """What is the probability of a given set of scores producing the given
        list of permutations only considering the top `equiv_len` ranks?"""
        log_scores = scores if self.log_scores else torch.log(scores)
        s = self.select_indices(log_scores, action)
        n = log_scores.shape[-1]
        p = self.upto if self.upto is not None else n
        return -sum(
            torch.log(torch.exp((s[k:] - s[k]) * self.shape).sum(dim=0))
            for k in range(p)
        )
