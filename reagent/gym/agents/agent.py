#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Optional

import torch
from reagent.gym.policies.policy import Policy
from reagent.gym.types import ActionPreprocessor, PostStep


def no_op(*args, **kwargs):
    pass


class Agent:
    def __init__(
        self,
        policy: Policy,
        action_preprocessor: ActionPreprocessor,
        post_transition_callback: PostStep = no_op,
    ):
        """
        The Agent orchestrates the interactions on our RL components, given
        the interactions with the environment.

        Args:
            policy: Policy that acts given preprocessed input
            action_preprocessor: preprocesses action for environment
            post_step: called after env.step(action).
                Default post_step is to do nothing.
        """
        self.policy = policy
        self.action_preprocessor = action_preprocessor
        self.post_transition_callback = post_transition_callback
        self._reset_internal_states()

    def _reset_internal_states(self):
        # intermediate state between act and post_step
        self._obs: Any = None
        self._actor_output: Any = None
        self._possible_actions_mask = None

    def act(
        self, obs: Any, possible_actions_mask: Optional[torch.Tensor] = None
    ) -> Any:
        actor_output = self.policy.act(obs, possible_actions_mask)

        # store intermediate states for post_step
        self._obs = obs
        self._actor_output = actor_output
        self._possible_actions_mask = possible_actions_mask

        # return action for the environment
        return self.action_preprocessor(actor_output)

    def post_step(self, reward: float, terminal: bool):
        """ to be called after step(action) """
        assert self._obs is not None
        assert self._actor_output is not None
        self.post_transition_callback(
            self._obs, self._actor_output, reward, terminal, self._possible_actions_mask
        )
        self._reset_internal_states()
