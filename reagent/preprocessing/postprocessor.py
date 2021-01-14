#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Dict, Tuple

import torch
import torch.nn as nn
from reagent.parameters import NormalizationParameters
from reagent.preprocessing.identify_types import (
    CONTINUOUS_ACTION,
    DISCRETE_ACTION,
    DO_NOT_PREPROCESS,
)
from reagent.preprocessing.normalization import EPS, get_num_output_features


class Postprocessor(nn.Module):
    """
    Inverting action
    """

    def __init__(
        self,
        normalization_parameters: Dict[int, NormalizationParameters],
        use_gpu: bool,
    ) -> None:
        super().__init__()
        self.num_output_features = get_num_output_features(normalization_parameters)
        feature_types = {
            norm_param.feature_type for norm_param in normalization_parameters.values()
        }
        assert (
            len(feature_types) == 1
        ), "All dimensions of actions should have the same preprocessing"
        self.feature_type = list(feature_types)[0]
        assert self.feature_type in {
            DISCRETE_ACTION,
            CONTINUOUS_ACTION,
            DO_NOT_PREPROCESS,
        }, f"{self.feature_type} is not DISCRETE_ACTION, CONTINUOUS_ACTION or DO_NOT_PREPROCESS"

        self.device = torch.device("cuda" if use_gpu else "cpu")

        if self.feature_type == CONTINUOUS_ACTION:
            sorted_features = sorted(normalization_parameters.keys())
            self.min_serving_value = torch.tensor(
                [normalization_parameters[f].min_value for f in sorted_features],
                device=self.device,
            ).float()
            self.scaling_factor = torch.tensor(
                [
                    (
                        # pyre-fixme[58]: `-` is not supported for operand types
                        #  `Optional[float]` and `Optional[float]`.
                        normalization_parameters[f].max_value
                        - normalization_parameters[f].min_value
                    )
                    / (2 * (1 - EPS))
                    for f in sorted_features
                ],
                device=self.device,
            ).float()
            self.almost_one = torch.tensor(1.0 - EPS, device=self.device).float()

    def input_prototype(self) -> Tuple[torch.Tensor]:
        return (torch.randn(1, self.num_output_features),)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.feature_type == CONTINUOUS_ACTION:
            # Please don't re-order; ONNX messed up tensor type when torch.clamp is
            # the first operand.
            return (
                self.almost_one + torch.clamp(input, -self.almost_one, self.almost_one)
            ) * self.scaling_factor + self.min_serving_value
        return input
