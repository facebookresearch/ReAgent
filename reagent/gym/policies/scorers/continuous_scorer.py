#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import reagent.types as rlt
import torch
from reagent.gym.types import GaussianSamplerScore, Scorer
from reagent.models.base import ModelBase


def sac_scorer(actor_network: ModelBase) -> Scorer:
    @torch.no_grad()
    def score(preprocessed_obs: rlt.PreprocessedState) -> GaussianSamplerScore:
        actor_network.eval()
        # TODO(kaiwenw) currently actor network demands a batched input.
        # should we make it single?
        state = rlt.PreprocessedFeatureVector(
            float_features=preprocessed_obs.state.float_features.unsqueeze(0)
        )
        loc, scale_log = actor_network._get_loc_and_scale_log(state)
        actor_network.train()
        return GaussianSamplerScore(loc=loc, scale_log=scale_log)

    return score
