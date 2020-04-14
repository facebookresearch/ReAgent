#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Optional

import reagent.types as rlt
import torch
from reagent.gym.types import PolicyPreprocessor, Sampler, Scorer


class Policy:
    def __init__(
        self, scorer: Scorer, sampler: Sampler, policy_preprocessor: PolicyPreprocessor
    ):
        """
        The Policy composes the scorer and sampler to create actions.

        Args:
            scorer: given preprocessed input, outputs intermediate scores
                used for sampling actions
            sampler: given scores (from the scorer), samples an action.
        """
        self.scorer = scorer
        self.sampler = sampler
        self.policy_preprocessor = policy_preprocessor

    def act(
        self, obs: Any, possible_actions_mask: Optional[torch.Tensor] = None
    ) -> rlt.ActorOutput:
        """
        Performs the composition described above.
        Optionally takes in a possible_actions_mask
            (only useful in the discrete case)
        These are the actions being put into the replay buffer, not necessary
        the actions taken by the environment! Those will be preprocessed once more
        with the action_preprocessor.
        """
        preprocessed_obs = self.policy_preprocessor(obs)
        scores = self.scorer(preprocessed_obs)
        if possible_actions_mask is None:
            # samplers that don't expect this mask will go here
            actor_output = self.sampler.sample_action(scores)
        else:
            actor_output = self.sampler.sample_action(scores, possible_actions_mask)

        # detach + convert to cpu
        actor_output.action = actor_output.action.cpu().detach()
        if actor_output.log_prob:
            actor_output.log_prob = actor_output.log_prob.cpu().detach()
        if actor_output.action_mean:
            actor_output.action_mean = actor_output.action_mean.cpu().detach()
        return actor_output
