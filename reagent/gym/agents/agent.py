#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Any, Optional, Union

import numpy as np
import reagent.types as rlt
import torch
from gym import Env, spaces
from reagent.gym.policies.policy import Policy
from reagent.gym.types import PostStep


def no_op(*args, **kwargs):
    pass


def _id(x):
    return x


class Agent:
    def __init__(
        self,
        policy: Policy,
        post_transition_callback: PostStep = no_op,
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

        if obs_preprocessor is not None:
            # Used whatever passed in, in case we want to apply normalization, etc.
            pass
        elif isinstance(observation_space, spaces.Box):

            # Maybe we need to organize the code better here
            def obs_preprocessor(obs: np.array) -> rlt.PreprocessedState:
                obs_tensor = torch.tensor(obs).float().unsqueeze(0)
                if device:
                    obs_tensor.to(device)
                return rlt.PreprocessedState.from_tensor(obs_tensor)

        else:
            raise NotImplementedError(
                f"Unsupport observation space: {observation_space}"
            )

        action_space = env.action_space

        if action_extractor is not None:
            pass
        elif isinstance(action_space, spaces.Discrete):

            def action_extractor(actor_output: rlt.ActorOutput):
                action = actor_output.action.squeeze(0)
                assert action.ndim == 1
                idx = action.argmax().numpy()
                return idx

        elif isinstance(action_space, spaces.Box):

            def action_extractor(actor_output: rlt.ActorOutput):
                action = actor_output.action.squeeze(0)
                assert action.ndim == 1 and action.shape == action_space.shape
                return action.numpy()

        else:
            raise NotImplementedError(f"Unsupport action space: {action_space}")

        return cls(
            policy,
            obs_preprocessor=obs_preprocessor,
            action_extractor=action_extractor,
            **kwargs,
        )

    def act(
        self, obs: Any, possible_actions_mask: Optional[torch.Tensor] = None
    ) -> Any:
        preprocessed_obs = self.obs_preprocessor(obs)

        if possible_actions_mask is not None:
            assert possible_actions_mask.ndim() == 1
            possible_actions_mask = possible_actions_mask.unsqueeze(0)

        actor_output = self.policy.act(preprocessed_obs, possible_actions_mask)

        # store intermediate states for post_step
        self._obs = obs
        self._actor_output = actor_output.squeeze(0)
        self._possible_actions_mask = (
            possible_actions_mask.squeeze(0)
            if possible_actions_mask is not None
            else None
        )

        return self.action_extractor(actor_output)

    def post_step(self, reward: float, terminal: bool):
        """ to be called after step(action) """
        assert self._obs is not None
        assert self._actor_output is not None
        self.post_transition_callback(
            self._obs, self._actor_output, reward, terminal, self._possible_actions_mask
        )
        self._reset_internal_states()
