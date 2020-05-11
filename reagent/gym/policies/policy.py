#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Optional

import reagent.types as rlt
import torch
from reagent.gym.types import Sampler, Scorer


class Policy:
    def __init__(self, scorer: Scorer, sampler: Sampler):
        """
        The Policy composes the scorer and sampler to create actions.

        Args:
            scorer: given preprocessed input, outputs intermediate scores
                used for sampling actions
            sampler: given scores (from the scorer), samples an action.
        """
        self.scorer = scorer
        self.sampler = sampler

    def act(
        self, obs: Any, possible_actions_mask: Optional[torch.Tensor] = None
    ) -> rlt.ActorOutput:
        """
        Performs the composition described above.
        Optionally takes in a possible_actions_mask
            (only useful in the discrete case)
        These are the actions being put into the replay buffer, not necessary
        the actions taken by the environment!
        """
        scores = self.scorer(obs)
        if possible_actions_mask is None:
            # samplers that don't expect this mask will go here
            # pyre-fixme[20]: Argument `possible_action_mask` expected.
            actor_output = self.sampler.sample_action(scores)
        else:
            actor_output = self.sampler.sample_action(scores, possible_actions_mask)

        return actor_output.cpu().detach()
