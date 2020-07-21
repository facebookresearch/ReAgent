#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from reagent.gym.envs.env_wrapper import EnvWrapper
from reagent.gym.policies.policy import Policy
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

    @classmethod
    def create_for_env(
        cls,
        env: EnvWrapper,
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
            obs_preprocessor = env.get_obs_preprocessor(device=device)

        if action_extractor is None:
            action_extractor = env.get_action_extractor()

        return cls(
            policy,
            obs_preprocessor=obs_preprocessor,
            action_extractor=action_extractor,
            **kwargs,
        )

    @classmethod
    def create_for_env_with_serving_policy(
        cls,
        env: EnvWrapper,
        serving_policy: Policy,
        *,
        obs_preprocessor=None,
        action_extractor=None,
        **kwargs,
    ):
        # device shouldn't be provided as serving is CPU only
        if obs_preprocessor is None:
            obs_preprocessor = env.get_serving_obs_preprocessor()

        if action_extractor is None:
            action_extractor = env.get_serving_action_extractor()

        return cls(
            serving_policy,
            obs_preprocessor=obs_preprocessor,
            action_extractor=action_extractor,
            **kwargs,
        )

    def act(
        self, obs: Any, possible_actions_mask: Optional[np.ndarray] = None
    ) -> Tuple[Any, Optional[float]]:
        """ Act on a single observation """
        # preprocess and convert to batch data
        preprocessed_obs = self.obs_preprocessor(obs)

        # store intermediate actor output for post_step
        actor_output = self.policy.act(preprocessed_obs, possible_actions_mask)
        log_prob = actor_output.log_prob
        if log_prob is not None:
            log_prob = log_prob.cpu().squeeze(0).item()
        return self.action_extractor(actor_output), log_prob

    def post_step(self, transition: Transition):
        """ to be called after step(action) """
        if self.post_transition_callback is not None:
            # pyre-fixme[29]: `Optional[typing.Callable[[Transition], None]]` is not
            #  a function.
            self.post_transition_callback(transition)
