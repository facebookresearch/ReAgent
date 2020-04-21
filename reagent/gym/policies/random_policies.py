#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Optional

import reagent.types as rlt
import torch
import torch.nn.functional as F
from reagent.gym.policies.policy import Policy


class DiscreteRandomPolicy(Policy):
    def __init__(self, num_actions):
        """ Random actor for accumulating random offline data. """
        self.num_actions = num_actions
        self.default_weights = torch.ones(num_actions)

    def act(
        self, obs: Any, possible_actions_mask: Optional[torch.Tensor] = None
    ) -> rlt.ActorOutput:
        """ Act randomly regardless of the observation. """
        weights = self.default_weights
        if possible_actions_mask:
            assert possible_actions_mask.shape == self.default_weights.shape
            weights = weights * possible_actions_mask

        # sample a random action
        m = torch.distributions.Categorical(weights)
        raw_action = m.sample()
        action = F.one_hot(raw_action, self.num_actions).squeeze(0)
        log_prob = m.log_prob(raw_action).float().squeeze(0)
        return rlt.ActorOutput(action=action, log_prob=log_prob)
