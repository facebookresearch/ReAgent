#!/usr/bin/env python3

import abc

import torch
from reagent.core.registry_meta import RegistryMeta


class SlateRankingNetBuilder:
    """
    Base class for slate ranking network builder.
    """

    @abc.abstractmethod
    def build_slate_ranking_network(
        self, state_dim, candidate_dim, candidate_size, slate_size
    ) -> torch.nn.Module:
        pass
