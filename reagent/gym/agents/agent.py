#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import inspect
from typing import Any, Optional, Union

import numpy as np
import reagent.types as rlt
import torch
from gym import Env, spaces
from reagent.gym.policies.policy import Policy
from reagent.gym.types import PostStep


def _id(x):
    return x


""" Default obs preprocessors.
    These should operate on single obs.
"""


def box_obs_preprocessor(obs: np.array) -> rlt.PreprocessedState:
    obs_tensor = torch.tensor(obs).float()
    return rlt.PreprocessedState.from_tensor(obs_tensor).unsqueeze(0)


""" Default action extractors.
    These currently operate on single action.
"""


def discrete_action_extractor(actor_output: rlt.ActorOutput):
    action = actor_output.action
    assert (
        action.ndim == 2 and action.shape[0] == 1
    ), f"{action} is not a single batch of results!"
    return action.squeeze(0).argmax().numpy()


def rescale_actions(
    actions: np.ndarray,
    new_min: float,
    new_max: float,
    prev_min: float,
    prev_max: float,
):
    """ Scale from [prev_min, prev_max] to [new_min, new_max] """
    prev_range = prev_max - prev_min
    new_range = new_max - new_min
    return ((actions - prev_min) / prev_range) * new_range + new_min


def make_box_action_extractor(action_space: spaces.Box):
    assert (
        len(action_space.shape) == 1 and action_space.shape[0] == 1
    ), f"{action_space} not supported."

    # NOTE: assuming actions are in range (-1, 1) and
    # need to be rescaled to the environment
    model_low = -1.0 + 1e-6
    model_high = +1.0 - 1e-6
    env_low = action_space.low.item()
    env_high = action_space.high.item()

    def box_action_extractor(actor_output: rlt.ActorOutput):
        action = actor_output.action
        assert (
            action.ndim == 2 and action.shape[0] == 1
        ), f"{action} is not a single batch of results!"
        return rescale_actions(
            action.squeeze(0).numpy(),
            new_min=env_low,
            new_max=env_high,
            prev_min=model_low,
            prev_max=model_high,
        )

    return box_action_extractor


class Agent:
    def __init__(
        self,
        policy: Policy,
        post_transition_callback: Optional[PostStep] = None,
        device: Optional[Union[str, torch.device]] = None,
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

        if device is not None and isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # check if policy has possible_actions_mask params (cts don't)
        sig = inspect.signature(policy.act)
        # Assuming state is first parameter and possible_actions_mask is second
        self.pass_in_pam = "possible_actions_mask" in sig.parameters
        if not self.pass_in_pam:
            assert len(sig.parameters) == 1, (
                f"{sig.parameters} has length other than 1, "
                "despite not having possible_actions_mask"
            )

    def _reset_internal_states(self):
        # intermediate state between act and post_step
        self._obs: Any = None
        self._actor_output: Any = None
        self._possible_actions_mask = None

    @classmethod
    def create_for_env(
        cls,
        env: Env,
        policy: Policy,
        *,
        device: Optional[Union[str, torch.device]] = None,
        obs_preprocessor=None,
        action_extractor=None,
        **kwargs,
    ):
        if device is not None and isinstance(device, str):
            device = torch.device(device)

        observation_space = env.observation_space

        if obs_preprocessor is None:
            # make default obs_preprocessors
            if isinstance(observation_space, spaces.Box):
                obs_preprocessor = box_obs_preprocessor
            else:
                raise NotImplementedError(
                    f"Unsupport observation space: {observation_space}"
                )

        action_space = env.action_space

        if action_extractor is None:
            # make default action_extractors
            if isinstance(action_space, spaces.Discrete):
                action_extractor = discrete_action_extractor
            elif isinstance(action_space, spaces.Box):
                action_extractor = make_box_action_extractor(action_space)
            else:
                raise NotImplementedError(f"Unsupport action space: {action_space}")

        return cls(
            policy,
            obs_preprocessor=obs_preprocessor,
            action_extractor=action_extractor,
            device=device,
            **kwargs,
        )

    def act(
        self, obs: Any, possible_actions_mask: Optional[torch.Tensor] = None
    ) -> Any:
        """ Act on a single observation """

        # store intermediate obs and possible_actions_mask for post_step
        self._obs = obs
        self._possible_actions_mask = possible_actions_mask

        # preprocess and convert to batch data
        preprocessed_obs = self.obs_preprocessor(obs)
        if self.device is not None:
            preprocessed_obs = preprocessed_obs.to(self.device)

        # optionally feed possible_actions_mask
        if self.pass_in_pam:
            if possible_actions_mask is not None:
                assert possible_actions_mask.ndim() == 1
                possible_actions_mask = possible_actions_mask.unsqueeze(0)
                if self.device is not None:
                    possible_actions_mask = possible_actions_mask.to(self.device)
            actor_output = self.policy.act(preprocessed_obs, possible_actions_mask)
        else:
            assert possible_actions_mask is None
            actor_output = self.policy.act(preprocessed_obs)

        # store intermediate actor output for post_step
        # NOTE: it is critical we store the actor output and not the
        # action taken by the environment, which may be normalized or processed.
        # E.g. SAC requires input actions to be scaled (-1, 1).
        # E.g. DiscreteDQN expects one-hot encoded
        self._actor_output = actor_output.squeeze(0)

        return self.action_extractor(actor_output)

    def post_step(self, reward: float, terminal: bool):
        """ to be called after step(action) """
        assert self._obs is not None
        assert self._actor_output is not None
        if self.post_transition_callback is not None:
            self.post_transition_callback(
                self._obs,
                self._actor_output,
                reward,
                terminal,
                self._possible_actions_mask,
            )
        self._reset_internal_states()
