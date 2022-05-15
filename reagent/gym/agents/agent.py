#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from reagent.gym.envs.env_wrapper import EnvWrapper
from reagent.gym.policies.policy import Policy
from reagent.gym.policies.random_policies import make_random_policy_for_env
from reagent.gym.types import PostEpisode, PostStep, Trajectory, Transition


def _id(x):
    return x


class Agent:
    def __init__(
        self,
        policy: Policy,
        post_transition_callback: Optional[PostStep] = None,
        post_episode_callback: Optional[PostEpisode] = None,
        obs_preprocessor=_id,
        action_extractor=_id,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        The Agent orchestrates the interactions on our RL components, given
        the interactions with the environment.

        Args:
            policy: Policy that acts given preprocessed input
            action_preprocessor: preprocesses action for environment
            post_step: called after env.step(action).
                Default post_step is to do nothing.
        """
        device = device or torch.device("cpu")
        self.policy = policy
        self.obs_preprocessor = obs_preprocessor
        self.action_extractor = action_extractor
        self.post_transition_callback = post_transition_callback
        self.post_episode_callback = post_episode_callback
        self.device = device

    @classmethod
    def create_for_env(
        cls,
        env: EnvWrapper,
        policy: Optional[Policy],
        *,
        device: Union[str, torch.device] = "cpu",
        obs_preprocessor=None,
        action_extractor=None,
        **kwargs
    ) -> "Agent":
        """
        If `policy` is not given, we will try to create a random policy
        """
        if isinstance(device, str):
            device = torch.device(device)

        if obs_preprocessor is None:
            obs_preprocessor = env.get_obs_preprocessor(device=device)

        if action_extractor is None:
            action_extractor = env.get_action_extractor()

        if policy is None:
            policy = make_random_policy_for_env(env)

        return cls(
            policy,
            obs_preprocessor=obs_preprocessor,
            action_extractor=action_extractor,
            device=device,
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
        **kwargs
    ) -> "Agent":
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
        """Act on a single observation"""
        # preprocess and convert to batch data
        preprocessed_obs = self.obs_preprocessor(obs)
        if possible_actions_mask is not None:
            possible_actions_mask = torch.tensor(
                possible_actions_mask, device=self.device
            )

        # store intermediate actor output for post_step
        actor_output = self.policy.act(preprocessed_obs, possible_actions_mask)
        log_prob = actor_output.log_prob
        if log_prob is not None:
            log_prob = log_prob.cpu().squeeze(0).item()
        return self.action_extractor(actor_output), log_prob

    def post_step(self, transition: Transition) -> None:
        """to be called after step(action)"""
        if self.post_transition_callback is not None:
            self.post_transition_callback(transition)

    def post_episode(self, trajectory: Trajectory, info: Dict) -> None:
        """to be called after step(action)"""
        if self.post_episode_callback is not None:
            self.post_episode_callback(trajectory, info)
