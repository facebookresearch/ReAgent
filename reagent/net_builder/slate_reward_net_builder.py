#!/usr/bin/env python3

import abc

import torch


class SlateRewardNetBuilder:
    """
    Base class for slate reward network builder.
    """

    @abc.abstractmethod
    def build_slate_reward_network(
        self, state_dim, candidate_dim, candidate_size, slate_size
    ) -> torch.nn.Module:
        pass

    @abc.abstractproperty
    def expect_slate_wise_reward(self) -> bool:
        pass
