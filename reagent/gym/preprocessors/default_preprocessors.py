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


logger = logging.getLogger(__name__)

#######################################
### Default obs preprocessors.
### These should operate on single obs.
#######################################


class RecsimObsPreprocessor:
    def __init__(
        self,
        *,
        num_docs: int,
        discrete_keys: List[Tuple[str, int]],
        box_keys: List[Tuple[str, int]],
    ):
        self.num_docs = num_docs
        self.discrete_keys = discrete_keys
        self.box_keys = box_keys

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
        return rlt.FeatureData(float_features=user, candidate_docs=candidate_docs)
