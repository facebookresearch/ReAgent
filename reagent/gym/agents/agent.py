#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import inspect
from typing import Any, Optional, Union

import numpy as np
import torch
from gym import Env
from reagent.gym.policies.policy import Policy
from reagent.gym.preprocessors import (
    make_default_action_extractor,
    make_default_obs_preprocessor,
    make_default_serving_action_extractor,
    make_default_serving_obs_preprocessor,
)
from reagent.gym.types import PostStep


def _id(x):
    return x


def to_device(obs: Any, device: torch.device):
    if isinstance(obs, np.ndarray):
        return torch.tensor(obs).to(device, non_blocking=True)
    elif isinstance(obs, dict):
        out = {}
        for k in obs:
            out[k] = obs[k].to(device, non_blocking=True)
        return out
    else:
        raise NotImplementedError(f"obs of type {type(obs)} not supported.")


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

        # check if policy.act needs possible_actions_mask (continuous policies don't)
        sig = inspect.signature(policy.act)
        # Assuming state is first parameter and possible_actions_mask is second
        self.pass_in_possible_actions_mask = "possible_actions_mask" in sig.parameters
        if not self.pass_in_possible_actions_mask and len(sig.parameters) != 1:
            raise RuntimeError(
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
        device: Union[str, torch.device] = "cpu",
        obs_preprocessor=None,
        action_extractor=None,
        **kwargs,
    ):
        if isinstance(device, str):
            device = torch.device(device)

        if obs_preprocessor is None:
            obs_preprocessor = make_default_obs_preprocessor(env)

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
    def create_from_serving_policy(cls, serving_policy, env: Env, **kwargs):
        obs_preprocessor = make_default_serving_obs_preprocessor(env)
        action_extractor = make_default_serving_action_extractor(env)
        return cls(
            serving_policy,
            obs_preprocessor=obs_preprocessor,
            action_extractor=action_extractor,
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
        device_obs = to_device(obs, self.device)
        preprocessed_obs = self.obs_preprocessor(device_obs)

        # optionally feed possible_actions_mask
        if self.pass_in_possible_actions_mask:
            # if possible_actions_mask is given, convert to batch of one
            # NOTE: it's still possible that possible_actions_mask is None
            if possible_actions_mask is not None:
                # pyre-fixme[16]: `Tensor` has no attribute `ndim`.
                assert possible_actions_mask.ndim() == 1
                possible_actions_mask = possible_actions_mask.unsqueeze(0).to(
                    self.device, non_blocking=True
                )
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
            # pyre-fixme[29]: `Optional[typing.Callable[[typing.Any,
            #  reagent.types.ActorOutput, float, bool, Optional[torch.Tensor]], None]]`
            #  is not a function.
            self.post_transition_callback(
                self._obs,
                self._actor_output,
                reward,
                terminal,
                self._possible_actions_mask,
            )
        self._reset_internal_states()
