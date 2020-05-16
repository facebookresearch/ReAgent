#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import reagent.types as rlt
import torch
from reagent.gym.types import Sampler


class TopKSampler(Sampler):
    def __init__(self, k: int):
        self.k = k

    def sample_action(self, scores: torch.Tensor) -> rlt.ActorOutput:
        top_values, item_idxs = torch.topk(scores, self.k, dim=1)
        return rlt.ActorOutput(
            action=item_idxs, log_prob=torch.zeros(item_idxs.shape[0], 1)
        )

    def log_prob(self, scores: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
