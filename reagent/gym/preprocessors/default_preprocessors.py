#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

""" Get default preprocessors for training time. """

import logging
from typing import List, Optional, Tuple

import numpy as np
import reagent.types as rlt
import torch
import torch.nn.functional as F
from gym import Env, spaces
from reagent.parameters import CONTINUOUS_TRAINING_ACTION_RANGE
from reagent.training.utils import rescale_actions


logger = logging.getLogger(__name__)

try:
    from recsim.simulator.recsim_gym import RecSimGymEnv

    HAS_RECSIM = True
except ImportError:
    HAS_RECSIM = False
    logger.warning(f"ReplayBuffer.create_from_env() will not recognize RecSim env")


def make_default_obs_preprocessor(env: Env, *, device: Optional[torch.device] = None):
    """ Returns the default obs preprocessor for the environment """
    if device is None:
        device = torch.device("cpu")
    observation_space = env.observation_space
    if HAS_RECSIM and isinstance(env.unwrapped, RecSimGymEnv):
        return RecsimObsPreprocessor.create_from_env(env, device=device)
    elif isinstance(observation_space, spaces.Box):
        return BoxObsPreprocessor(device)
    else:
        raise NotImplementedError(f"Unsupport observation space: {observation_space}")


def make_default_action_extractor(env: Env):
    """ Returns the default action extractor for the environment """
    action_space = env.action_space
    if isinstance(action_space, spaces.Discrete):
        # Canonical rule to return one-hot encoded actions for discrete
        return discrete_action_extractor
    elif isinstance(action_space, spaces.MultiDiscrete):
        return multi_discrete_action_extractor
    elif isinstance(action_space, spaces.Box):
        # Canonical rule to scale actions to CONTINUOUS_TRAINING_ACTION_RANGE
        return make_box_action_extractor(action_space)
    else:
        raise NotImplementedError(f"Unsupport action space: {action_space}")


#######################################
### Default obs preprocessors.
### These should operate on single obs.
#######################################
class BoxObsPreprocessor:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, obs: np.ndarray) -> rlt.FeatureData:
        return rlt.FeatureData(torch.tensor(obs).float().unsqueeze(0)).to(
            self.device, non_blocking=True
        )


class RecsimObsPreprocessor:
    def __init__(
        self,
        *,
        num_docs: int,
        discrete_keys: List[Tuple[str, int]],
        box_keys: List[Tuple[str, int]],
        device: torch.device,
    ):
        self.num_docs = num_docs
        self.discrete_keys = discrete_keys
        self.box_keys = box_keys
        self.device = device

    @classmethod
    def create_from_env(cls, env: Env, **kwargs):
        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Dict)
        user_obs_space = obs_space["user"]
        if not isinstance(user_obs_space, spaces.Box):
            raise NotImplementedError(
                f"User observation space {type(user_obs_space)} is not supported"
            )

        doc_obs_space = obs_space["doc"]
        if not isinstance(doc_obs_space, spaces.Dict):
            raise NotImplementedError(
                f"Doc space {type(doc_obs_space)} is not supported"
            )

        # Assume that all docs are in the same space

        discrete_keys: List[Tuple[str, int]] = []
        box_keys: List[Tuple[str, int]] = []

        doc_0_space = doc_obs_space["0"]

        if isinstance(doc_0_space, spaces.Dict):
            for k, v in doc_obs_space["0"].spaces.items():
                if isinstance(v, spaces.Discrete):
                    if v.n > 0:
                        discrete_keys.append((k, v.n))
                elif isinstance(v, spaces.Box):
                    shape_dim = len(v.shape)
                    if shape_dim == 0:
                        box_keys.append((k, 1))
                    elif shape_dim == 1:
                        box_keys.append((k, v.shape[0]))
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError(
                        f"Doc feature {k} with the observation space of {type(v)}"
                        " is not supported"
                    )
        elif isinstance(doc_0_space, spaces.Box):
            pass
        else:
            raise NotImplementedError(f"Unknown space {doc_0_space}")

        return cls(
            num_docs=len(doc_obs_space.spaces),
            discrete_keys=sorted(discrete_keys),
            box_keys=sorted(box_keys),
            **kwargs,
        )

    def __call__(self, obs):
        user = torch.tensor(obs["user"]).float().unsqueeze(0)

        doc_obs = obs["doc"]

        if self.discrete_keys or self.box_keys:
            # Dict space
            discrete_features: List[torch.Tensor] = []
            for k, n in self.discrete_keys:
                vals = torch.tensor([v[k] for v in doc_obs.values()])
                assert vals.shape == (self.num_docs,)
                discrete_features.append(F.one_hot(vals, n).float())

            box_features: List[torch.Tensor] = []
            for k, d in self.box_keys:
                vals = np.vstack([v[k] for v in doc_obs.values()])
                assert vals.shape == (self.num_docs, d)
                box_features.append(torch.tensor(vals).float())

            doc_features = torch.cat(discrete_features + box_features, dim=1).unsqueeze(
                0
            )
        else:
            # Simply a Box space
            vals = np.vstack(list(doc_obs.values()))
            doc_features = torch.tensor(vals).float().unsqueeze(0)

        # This comes from ValueWrapper
        value = (
            torch.tensor([v["value"] for v in obs["augmentation"].values()])
            .float()
            .unsqueeze(0)
        )

        candidate_docs = rlt.DocList(
            float_features=doc_features,
            mask=torch.ones(doc_features.shape[:-1], dtype=torch.bool),
            value=value,
        )
        return rlt.FeatureData(float_features=user, candidate_docs=candidate_docs).to(
            self.device, non_blocking=True
        )


############################################
### Default action extractors.
### These currently operate on single action.
############################################
def discrete_action_extractor(actor_output: rlt.ActorOutput):
    action = actor_output.action
    assert (
        # pyre-fixme[16]: `Tensor` has no attribute `ndim`.
        action.ndim == 2
        and action.shape[0] == 1
    ), f"{action} is not a single batch of results!"
    # pyre-fixme[16]: `Tensor` has no attribute `argmax`.
    return action.squeeze(0).argmax().cpu().numpy()


def multi_discrete_action_extractor(actor_output: rlt.ActorOutput):
    return actor_output.action.squeeze(0).cpu().numpy()


def make_box_action_extractor(action_space: spaces.Box):
    assert len(action_space.shape) == 1, f"{action_space} not supported."

    model_low, model_high = CONTINUOUS_TRAINING_ACTION_RANGE

    def box_action_extractor(actor_output: rlt.ActorOutput) -> np.ndarray:
        action = actor_output.action
        assert (
            len(action.shape) == 2 and action.shape[0] == 1
        ), f"{action} (shape: {action.shape}) is not a single action!"
        return rescale_actions(
            action.squeeze(0).cpu().numpy(),
            new_min=action_space.low,
            new_max=action_space.high,
            prev_min=model_low,
            prev_max=model_high,
        )

    return box_action_extractor
