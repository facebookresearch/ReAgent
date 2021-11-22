#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
from reagent.preprocessing.identify_types import CONTINUOUS, CONTINUOUS_ACTION
from reagent.preprocessing.normalization import NormalizationParameters


def _cont_norm():
    return NormalizationParameters(feature_type=CONTINUOUS, mean=0.0, stddev=1.0)


def _cont_action_norm():
    return NormalizationParameters(
        feature_type=CONTINUOUS_ACTION, min_value=-3.0, max_value=3.0
    )


def change_cand_size_slate_ranking(input_prototype, candidate_size_override):
    state_prototype, candidate_prototype = input_prototype
    candidate_prototype = (
        candidate_prototype[0][:, :1, :].repeat(1, candidate_size_override, 1),
        candidate_prototype[1][:, :1, :].repeat(1, candidate_size_override, 1),
    )
    return (
        (torch.randn_like(state_prototype[0]), torch.ones_like(state_prototype[1])),
        (
            torch.randn_like(candidate_prototype[0]),
            torch.ones_like(candidate_prototype[1]),
        ),
    )
