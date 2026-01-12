#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe
import logging
import math
from typing import Optional

import reagent.core.types as rlt
import torch
import torch.nn.functional as F
from reagent.core.configuration import resolve_defaults
from reagent.gym.types import Sampler
from torch.distributions import Gumbel

logger = logging.getLogger(__name__)


class FrechetSort(Sampler):
    EPS = 1e-12

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
            in probability computations.
            Essentially specifies the action space.
        :param log_scores Scores passed in are already log-transformed. In this case, we would
        simply add Gumbel noise.
        For LearnVM, we set this to be True because we expect input and output scores
        to be in the log space.

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

    def sample_action(self, scores: torch.Tensor) -> rlt.ActorOutput:
        """Sample a ranking according to Frechet sort. Note that possible_actions_mask
        is ignored as the list of rankings scales exponentially with slate size and
        number of items and it can be difficult to enumerate them."""
        assert scores.dim() == 2, "sample_action only accepts batches"
        log_scores = scores if self.log_scores else torch.log(scores)
        perturbed = log_scores + self.gumbel_noise.sample(scores.shape)
        action = torch.argsort(perturbed.detach(), descending=True)
        log_prob = self.log_prob(scores, action)
        # Only truncate the action before returning
        if self.topk is not None:
            action = action[: self.topk]
        return rlt.ActorOutput(action, log_prob)

    def log_prob(
        self,
        scores: torch.Tensor,
        action: torch.Tensor,
        equiv_len_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        What is the probability of a given set of scores producing the given
        list of permutations only considering the top `equiv_len` ranks?

        We may want to override the default equiv_len here when we know the having larger
        action space doesn't matter. i.e. in Reels
        """
        upto = self.upto
        if equiv_len_override is not None:
            assert equiv_len_override.shape == (scores.shape[0],), (
                f"Invalid shape {equiv_len_override.shape}, compared to scores {scores.shape}. equiv_len_override {equiv_len_override}"
            )
            upto = equiv_len_override.long()
            if self.topk is not None and torch.any(equiv_len_override > self.topk):
                raise ValueError(
                    f"Override {equiv_len_override} cannot exceed topk={self.topk}."
                )

        squeeze = False
        if len(scores.shape) == 1:
            squeeze = True
            scores = scores.unsqueeze(0)
            action = action.unsqueeze(0)

        assert len(action.shape) == len(scores.shape) == 2, "scores should be batch"
        if action.shape[1] > scores.shape[1]:
            raise ValueError(
                f"action cardinality ({action.shape[1]}) is larger than the number of scores ({scores.shape[1]})"
            )
        elif action.shape[1] < scores.shape[1]:
            raise NotImplementedError(
                f"This semantic is ambiguous. If you have shorter slate, pad it with scores.shape[1] ({scores.shape[1]})"
            )

        log_scores = scores if self.log_scores else torch.log(scores)
        n = log_scores.shape[-1]
        # Add scores for the padding value
        log_scores = torch.cat(
            [
                log_scores,
                torch.full(
                    (log_scores.shape[0], 1), -math.inf, device=log_scores.device
                ),
            ],
            dim=1,
        )
        log_scores = torch.gather(log_scores, 1, action) * self.shape

        p = upto if upto is not None else n
        # We should unsqueeze here
        if isinstance(p, int):
            log_prob = sum(
                torch.nan_to_num(
                    F.log_softmax(log_scores[:, i:], dim=1)[:, 0], neginf=0.0
                )
                for i in range(p)
            )
        elif isinstance(p, torch.Tensor):
            # do masked sum
            log_prob = sum(
                torch.nan_to_num(
                    F.log_softmax(log_scores[:, i:], dim=1)[:, 0], neginf=0.0
                )
                * (i < p).float()
                for i in range(n)
            )
        else:
            raise RuntimeError(f"p is {p}")

        # pyre-fixme[16]: Item `int` of `Union[typing_extensions.Literal[0],
        #  Tensor]` has no attribute `isnan`.
        assert not torch.any(log_prob.isnan()), f"Nan in {log_prob}"
        # pyre-fixme[7]: Expected `Tensor` but got `Union[int, Tensor]`.
        return log_prob
