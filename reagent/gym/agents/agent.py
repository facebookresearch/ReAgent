#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Optional, Union

import torch
from gym import Env
from reagent.gym.policies.policy import Policy
from reagent.gym.preprocessors import (
    make_default_action_extractor,
    make_default_obs_preprocessor,
    make_default_serving_action_extractor,
    make_default_serving_obs_preprocessor,
)
from reagent.gym.types import PostStep, Transition


def _id(x):
    return x


class Agent:
    def __init__(
        self,
        policy: Policy,
        post_transition_callback: Optional[PostStep] = None,
        device: Union[str, torch.device] = "cpu",
        obs_preprocessor=_id,
        action_extractor=_id,
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
        self.obs_preprocessor = obs_preprocessor
        self.action_extractor = action_extractor
        self.post_transition_callback = post_transition_callback
        self._reset_internal_states()

        if isinstance(device, str):
            device = torch.device(device)
        self.device: torch.device = device

    def _reset_internal_states(self):
        # intermediate state between act and post_step
        self._log_prob: float = 0.0

    @classmethod
    def create_for_env(
        cls,
        env: Env,
        policy: Policy,
        *,
        device: Union[str, torch.device] = "cpu",
        obs_preprocessor=None,
        action_extractor=None,
        **kwargs,
    ):
        if isinstance(device, str):
            device = torch.device(device)

        if obs_preprocessor is None:
            obs_preprocessor = make_default_obs_preprocessor(env, device=device)

        if action_extractor is None:
            action_extractor = make_default_action_extractor(env)

        return cls(
            policy,
            obs_preprocessor=obs_preprocessor,
            action_extractor=action_extractor,
            device=device,
            **kwargs,
        )

    @classmethod
    def create_for_env_with_serving_policy(
        cls, env: Env, serving_policy: Policy, **kwargs
    ):
        obs_preprocessor = make_default_serving_obs_preprocessor(env)
        action_extractor = make_default_serving_action_extractor(env)
        return cls(
            serving_policy,
            obs_preprocessor=obs_preprocessor,
            action_extractor=action_extractor,
            **kwargs,
        )

    def act(self, obs: Any) -> Any:
        """ Act on a single observation """
        # preprocess and convert to batch data
        preprocessed_obs = self.obs_preprocessor(obs)

        # store intermediate actor output for post_step
        actor_output = self.policy.act(preprocessed_obs)
        self._log_prob = (
            0.0
            if actor_output.log_prob is None
            # pyre-fixme[16]: `Optional` has no attribute `cpu`.
            else actor_output.log_prob.cpu().squeeze(0).item()
        )
        return self.action_extractor(actor_output)

    def post_step(self, transition: Transition):
        """ to be called after step(action) """
        if self.post_transition_callback is not None:
            transition.log_prob = self._log_prob
            # pyre-fixme[29]: `Optional[typing.Callable[[Transition], None]]` is not
            #  a function.
            self.post_transition_callback(transition)
        self._reset_internal_states()
