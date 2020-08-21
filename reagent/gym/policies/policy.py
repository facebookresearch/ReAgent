#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Optional

import numpy as np
import reagent.types as rlt
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
        self, obs: Any, possible_actions_mask: Optional[np.ndarray] = None
    ) -> rlt.ActorOutput:
        """
        Performs the composition described above.
        These are the actions being put into the replay buffer, not necessary
        the actions taken by the environment!
        """
        scorer_inputs = (obs,)
        if possible_actions_mask is not None:
            scorer_inputs += (possible_actions_mask,)
        scores = self.scorer(*scorer_inputs)
        actor_output = self.sampler.sample_action(scores)
        return actor_output.cpu().detach()
