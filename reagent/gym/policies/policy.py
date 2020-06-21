#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import Any

import reagent.types as rlt
from reagent.gym.types import Sampler, Scorer


class Policy:
    def __init__(self, scorer: Scorer, sampler: Sampler, expand_args: bool = False):
        """
        The Policy composes the scorer and sampler to create actions.

        Args:
            scorer: given preprocessed input, outputs intermediate scores
                used for sampling actions
            sampler: given scores (from the scorer), samples an action.
        """
        self.scorer = scorer
        self.sampler = sampler
        # TODO: consolidate inputs of predictor networks to be FeatureData
        # then we won't need this
        self.expand_args = expand_args

    def act(self, obs: Any) -> rlt.ActorOutput:
        """
        Performs the composition described above.
        These are the actions being put into the replay buffer, not necessary
        the actions taken by the environment!
        """
        if self.expand_args:
            scores = self.scorer(*obs)
        else:
            scores = self.scorer(obs)
        actor_output = self.sampler.sample_action(scores)
        return actor_output.cpu().detach()
