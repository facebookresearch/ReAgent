#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import abc

import torch


class SlateRankingNetBuilder:
    """
    Base class for slate ranking network builder.
    """

    @abc.abstractmethod
    def build_slate_ranking_network(
        self, state_dim, candidate_dim, candidate_size, slate_size
    ) -> torch.nn.Module:
        pass
