#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import reagent.core.types as rlt
import torch
from reagent.gym.types import GaussianSamplerScore, Scorer
from reagent.models.base import ModelBase


def sac_scorer(actor_network: ModelBase) -> Scorer:
    @torch.no_grad()
    def score(preprocessed_obs: rlt.FeatureData) -> GaussianSamplerScore:
        actor_network.eval()
        # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
        loc, scale_log = actor_network._get_loc_and_scale_log(preprocessed_obs)
        actor_network.train()
        return GaussianSamplerScore(loc=loc, scale_log=scale_log)

    return score
