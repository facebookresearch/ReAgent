#!/usr/bin/env python3

import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from ml.rl.preprocessing import identify_types
from ml.rl.preprocessing.identify_types import FEATURE_TYPES
from ml.rl.preprocessing.normalization import (
    MISSING_VALUE,
    NormalizationParameters,
    get_num_output_features,
)


logger = logging.getLogger(__name__)


class Preprocessor(nn.Module):
    def __init__(
        self, normalization_parameters: Dict[str, NormalizationParameters]
    ) -> None:
        super(Preprocessor, self).__init__()
        self.normalization_parameters = normalization_parameters
        self.sorted_features, self.sorted_feature_boundaries = (
            self._sort_features_by_normalization()
        )
        self.clamp = True  # Only set to false in unit tests

    def forward(self, input) -> torch.FloatTensor:
        """ Preprocess the input matrix
        :param input tensor
        """
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)

        not_missing_input = (input != MISSING_VALUE).float()

        output_feature_dim = get_num_output_features(self.normalization_parameters)
        output = torch.zeros(input.shape[0], output_feature_dim)

        output_cursor = 0
        for column, feature in enumerate(self.sorted_features):
            norm_params = self.normalization_parameters[feature]
            output_size = (
                len(norm_params.possible_values)
                if norm_params.feature_type == identify_types.ENUM
                else 1
            )
            sliced_output = output[:, output_cursor : output_cursor + output_size]
            self._preprocess_feature(
                input[:, column].reshape(-1, 1), sliced_output, norm_params
            )
            output[:, output_cursor : output_cursor + output_size] *= not_missing_input[
                :, column
            ].reshape(-1, 1)
            output_cursor += output_size
        return output

    def _preprocess_feature(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        norm_params: NormalizationParameters,
    ) -> None:
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)

        feature_type = norm_params.feature_type
        func = getattr(self, "_preprocess_" + feature_type)
        func(input, output, norm_params)

    def _preprocess_BINARY(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        norm_params: NormalizationParameters,
    ) -> None:
        output[:] = (input != 0).float()

    def _preprocess_PROBABILITY(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        norm_params: NormalizationParameters,
    ) -> None:
        if self.clamp:
            clamped_input = torch.clamp(input, 0.01, 0.99)
        else:
            clamped_input = input
        output[:] = -1.0 * torch.log((1.0 / clamped_input) - 1.0)

    def _preprocess_CONTINUOUS(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        norm_params: NormalizationParameters,
    ) -> None:
        continuous_output = (input - norm_params.mean) / norm_params.stddev
        if not self.clamp:
            output[:] = continuous_output
        else:
            output[:] = torch.clamp(continuous_output, -3.0, 3.0)

    def _preprocess_BOXCOX(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        norm_params: NormalizationParameters,
    ) -> None:
        assert (
            norm_params.boxcox_lambda != 0
        ), "The estimated boxcox lambda should never be 0"
        boxcox_output = (
            torch.pow(input + norm_params.boxcox_shift, norm_params.boxcox_lambda) - 1.0
        ) / norm_params.boxcox_lambda
        boxcox_output = (boxcox_output - norm_params.mean) / norm_params.stddev
        if not self.clamp:
            output[:] = boxcox_output
        output[:] = torch.clamp(boxcox_output, -3.0, 3.0)

    def _preprocess_QUANTILE(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        norm_params: NormalizationParameters,
    ) -> None:
        num_quantiles = float(len(norm_params.quantiles)) - 1
        assert num_quantiles > 2, "Need at least two quantile boundaries (min and max)"
        quantile_values = torch.Tensor(norm_params.quantiles)
        max_quantile = float(np.max(norm_params.quantiles))
        min_quantile = float(np.min(norm_params.quantiles))
        set_to_min = (input <= min_quantile).float()
        set_to_max = (input >= max_quantile).float()
        interpolate = ((set_to_min + set_to_max) == 0).float()
        input_greater_than = (input >= quantile_values).float()
        interpolate_left, _ = torch.max(
            (input_greater_than * quantile_values)
            + ((input < quantile_values).float() * -1e20),
            dim=1,
            keepdim=True,
        )
        interpolate_right, _ = torch.min(
            ((input < quantile_values).float() * quantile_values)
            + ((input >= quantile_values).float() * 1e20),
            dim=1,
            keepdim=True,
        )
        interpolated_values = (
            (torch.sum(input_greater_than, dim=1, keepdim=True) - 1.0)
            + ((input - interpolate_left) / (interpolate_right - interpolate_left))
        ) / num_quantiles
        output[:] = set_to_max + (interpolate * interpolated_values)

    def _preprocess_ENUM(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        norm_params: NormalizationParameters,
    ) -> None:
        enum_values = torch.Tensor(norm_params.possible_values)
        output[:] = (input == enum_values).float()

    def _sort_features_by_normalization(self):
        """
        Helper function to return a sorted list from a normalization map.
        Also returns the starting index for each feature type"""
        # Sort features by feature type
        sorted_features = []
        feature_starts = []
        for feature_type in FEATURE_TYPES:
            feature_starts.append(len(sorted_features))
            for feature in self.normalization_parameters.keys():
                norm = self.normalization_parameters[feature]
                if norm.feature_type == feature_type:
                    sorted_features.append(feature)
        return sorted_features, feature_starts

    def _get_type_boundaries(
        self,
        features: List[str],
        normalization_parameters: Dict[str, NormalizationParameters],
    ) -> List[int]:
        feature_starts = []
        on_feature_type = -1
        for i, feature in enumerate(features):
            feature_type = normalization_parameters[feature].feature_type
            feature_type_index = FEATURE_TYPES.index(feature_type)
            assert (
                feature_type_index >= on_feature_type
            ), "Features are not sorted by feature type!"
            while feature_type_index > on_feature_type:
                feature_starts.append(i)
                on_feature_type += 1
        while on_feature_type < len(FEATURE_TYPES):
            feature_starts.append(len(features))
            on_feature_type += 1
        return feature_starts
