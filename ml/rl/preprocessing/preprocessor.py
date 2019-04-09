#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from ml.rl import types as rlt
from ml.rl.preprocessing.identify_types import ENUM, FEATURE_TYPES
from ml.rl.preprocessing.normalization import (
    EPS,
    MAX_FEATURE_VALUE,
    MIN_FEATURE_VALUE,
    MISSING_VALUE,
    NormalizationParameters,
)
from torch.nn import Module, Parameter


logger = logging.getLogger(__name__)


class Preprocessor(Module):
    def __init__(
        self,
        normalization_parameters: Dict[str, NormalizationParameters],
        use_gpu: bool,
        typed_output: bool = False,
    ) -> None:
        super().__init__()
        self.normalization_parameters = normalization_parameters
        self.sorted_features, self.sorted_feature_boundaries = (
            self._sort_features_by_normalization()
        )
        self.typed_output = typed_output

        cuda_available = torch.cuda.is_available()
        logger.info("CUDA availability: {}".format(cuda_available))
        if use_gpu and cuda_available:
            logger.info("Using GPU: GPU requested and available.")
            self.use_gpu = True
            self.dtype = torch.cuda.FloatTensor
        else:
            logger.info("NOT Using GPU: GPU not requested or not available.")
            self.use_gpu = False
            self.dtype = torch.FloatTensor

        # NOTE: Because of the way we call AppendNet to squash ONNX to a C2 net,
        # We need to make tensors for every numeric literal
        self.zero_tensor = Parameter(
            torch.tensor([0.0]).type(self.dtype), requires_grad=False
        )
        self.one_tensor = Parameter(
            torch.tensor([1.0]).type(self.dtype), requires_grad=False
        )
        self.one_half_tensor = Parameter(
            torch.tensor([0.5]).type(self.dtype), requires_grad=False
        )
        self.one_hundredth_tensor = Parameter(
            torch.tensor([0.01]).type(self.dtype), requires_grad=False
        )
        self.negative_one_tensor = Parameter(
            torch.tensor([-1.0]).type(self.dtype), requires_grad=False
        )
        self.missing_tensor = Parameter(
            torch.tensor([MISSING_VALUE]).type(self.dtype), requires_grad=False
        )
        self.min_tensor = Parameter(
            torch.tensor([-1e20]).type(self.dtype), requires_grad=False
        )
        self.max_tensor = Parameter(
            torch.tensor([1e20]).type(self.dtype), requires_grad=False
        )
        self.epsilon_tensor = Parameter(
            torch.tensor([EPS]).type(self.dtype), requires_grad=False
        )

        feature_starts = self._get_type_boundaries()
        for i, feature_type in enumerate(FEATURE_TYPES):
            begin_index = feature_starts[i]
            if (i + 1) == len(FEATURE_TYPES):
                end_index = len(self.normalization_parameters)
            else:
                end_index = feature_starts[i + 1]
            if begin_index == end_index:
                continue  # No features of this type
            if feature_type == ENUM:
                # Process one-at-a-time
                for j in range(begin_index, end_index):
                    norm_params = self.normalization_parameters[self.sorted_features[j]]
                    func = getattr(self, "_create_parameters_" + feature_type)
                    func(j, norm_params)
            else:
                norm_params = []
                for f in self.sorted_features[begin_index:end_index]:
                    norm_params.append(self.normalization_parameters[f])
                func = getattr(self, "_create_parameters_" + feature_type)
                func(begin_index, norm_params)

    def input_prototype(self):
        return rlt.FeatureVector(
            float_features=torch.randn(1, len(self.normalization_parameters))
        )

    def forward(self, input) -> torch.FloatTensor:
        """ Preprocess the input matrix
        :param input tensor
        """
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).type(self.dtype)
        if isinstance(input, rlt.FeatureVector):
            input = input.float_features.type(self.dtype)

        # ONNX doesn't support != yet
        not_missing_input = (
            self.one_tensor.float() - (input == self.missing_tensor).float()
        )
        feature_starts = self._get_type_boundaries()

        outputs = []
        for i, feature_type in enumerate(FEATURE_TYPES):
            begin_index = feature_starts[i]
            if (i + 1) == len(FEATURE_TYPES):
                end_index = len(self.normalization_parameters)
            else:
                end_index = feature_starts[i + 1]
            if begin_index == end_index:
                continue  # No features of this type
            if feature_type == ENUM:
                # Process one-at-a-time
                for j in range(begin_index, end_index):
                    norm_params = self.normalization_parameters[self.sorted_features[j]]
                    new_output = self._preprocess_feature_single_column(
                        j, input[:, j : j + 1], norm_params
                    )
                    new_output *= not_missing_input[:, j : j + 1]
                    self._check_preprocessing_output(new_output, [norm_params])
                    outputs.append(new_output)
            else:
                norm_params = []
                for f in self.sorted_features[begin_index:end_index]:
                    norm_params.append(self.normalization_parameters[f])
                new_output = self._preprocess_feature_multi_column(
                    begin_index, input[:, begin_index:end_index], norm_params
                )
                new_output *= not_missing_input[:, begin_index:end_index]
                self._check_preprocessing_output(new_output, norm_params)
                outputs.append(new_output)

        def wrap(output):
            if self.typed_output:
                return rlt.FeatureVector(float_features=output)
            else:
                return output

        if len(outputs) == 1:
            return wrap(torch.clamp(outputs[0], MIN_FEATURE_VALUE, MAX_FEATURE_VALUE))

        return wrap(
            torch.clamp(torch.cat(outputs, dim=1), MIN_FEATURE_VALUE, MAX_FEATURE_VALUE)
        )

    def _preprocess_feature_single_column(
        self,
        begin_index: int,
        input: torch.Tensor,
        norm_params: NormalizationParameters,
    ) -> torch.Tensor:
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).type(self.dtype)

        feature_type = norm_params.feature_type
        func = getattr(self, "_preprocess_" + feature_type)
        return func(begin_index, input, norm_params)

    def _preprocess_feature_multi_column(
        self,
        begin_index: int,
        input: torch.Tensor,
        norm_params: List[NormalizationParameters],
    ) -> torch.Tensor:
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input).type(self.dtype)

        feature_type = norm_params[0].feature_type
        func = getattr(self, "_preprocess_" + feature_type)
        return func(begin_index, input, norm_params)

    def _create_parameters_BINARY(
        self, begin_index: int, norm_params: List[NormalizationParameters]
    ):
        pass

    def _preprocess_BINARY(
        self,
        begin_index: int,
        input: torch.Tensor,
        norm_params: List[NormalizationParameters],
    ) -> torch.Tensor:
        # ONNX doesn't support != yet
        return self.one_tensor - (input == self.zero_tensor).float()

    def _create_parameters_PROBABILITY(
        self, begin_index: int, norm_params: List[NormalizationParameters]
    ):
        pass

    def _preprocess_PROBABILITY(
        self,
        begin_index: int,
        input: torch.Tensor,
        norm_params: List[NormalizationParameters],
    ) -> torch.Tensor:
        clamped_input = torch.clamp(input, 0.01, 0.99)
        return self.negative_one_tensor * (
            ((self.one_tensor / clamped_input) - self.one_tensor).log()
        )

    def _create_parameters_CONTINUOUS_ACTION(
        self, begin_index: int, norm_params: List[NormalizationParameters]
    ):
        self._create_parameter(
            begin_index,
            "min_serving_value",
            torch.Tensor([p.min_value for p in norm_params]).type(self.dtype),
        )
        self._create_parameter(
            begin_index,
            "min_training_value",
            torch.ones(len(norm_params)).type(self.dtype) * -1 + EPS,
        )
        self._create_parameter(
            begin_index,
            "scaling_factor",
            (torch.ones(len(norm_params)).type(self.dtype) - EPS)
            * 2
            / torch.tensor([p.max_value - p.min_value for p in norm_params]).type(
                self.dtype
            ),
        )

    def _preprocess_CONTINUOUS_ACTION(
        self,
        begin_index: int,
        input: torch.Tensor,
        norm_params: List[NormalizationParameters],
    ) -> torch.Tensor:
        min_serving_value = self._fetch_parameter(begin_index, "min_serving_value")
        min_training_value = self._fetch_parameter(begin_index, "min_training_value")
        scaling_factor = self._fetch_parameter(begin_index, "scaling_factor")
        continuous_action = (
            input - min_serving_value
        ) * scaling_factor + min_training_value
        return torch.clamp(continuous_action, -1 + EPS, 1 - EPS)

    def _create_parameters_CONTINUOUS(
        self, begin_index: int, norm_params: List[NormalizationParameters]
    ):
        self._create_parameter(
            begin_index,
            "means",
            torch.Tensor([p.mean for p in norm_params]).type(self.dtype),
        )
        self._create_parameter(
            begin_index,
            "stddevs",
            torch.Tensor([p.stddev for p in norm_params]).type(self.dtype),
        )

    def _preprocess_CONTINUOUS(
        self,
        begin_index: int,
        input: torch.Tensor,
        norm_params: List[NormalizationParameters],
    ) -> torch.Tensor:
        means = self._fetch_parameter(begin_index, "means")
        stddevs = self._fetch_parameter(begin_index, "stddevs")
        continuous_output = (input - means) / stddevs
        return torch.clamp(continuous_output, MIN_FEATURE_VALUE, MAX_FEATURE_VALUE)

    def _create_parameters_BOXCOX(
        self, begin_index: int, norm_params: List[NormalizationParameters]
    ):
        self._create_parameter(
            begin_index,
            "shifts",
            torch.Tensor([p.boxcox_shift for p in norm_params]).type(self.dtype),
        )
        for p in norm_params:
            assert (
                abs(p.boxcox_lambda) > 1e-6
            ), "Invalid value for boxcox lambda: " + str(p.boxcox_lambda)
        self._create_parameter(
            begin_index,
            "lambdas",
            torch.Tensor([p.boxcox_lambda for p in norm_params]).type(self.dtype),
        )
        self._create_parameters_CONTINUOUS(begin_index, norm_params)

    def _preprocess_BOXCOX(
        self,
        begin_index: int,
        input: torch.Tensor,
        norm_params: List[NormalizationParameters],
    ) -> torch.Tensor:
        shifts = self._fetch_parameter(begin_index, "shifts")
        lambdas = self._fetch_parameter(begin_index, "lambdas")
        boxcox_output = (
            # We can replace this with a normal pow() call after D8528654 lands
            self._manual_broadcast_matrix_scalar(
                torch.clamp(
                    input + shifts, 1e-6
                ),  # Clamp is necessary to prevent MISSING_VALUE from going to NaN
                lambdas,
                torch.pow,
            )
            - self.one_tensor
        ) / lambdas
        return self._preprocess_CONTINUOUS(begin_index, boxcox_output, norm_params)

    def _create_parameters_QUANTILE(
        self, begin_index: int, norm_params: List[NormalizationParameters]
    ):
        F = len(norm_params)

        num_quantiles = torch.tensor(
            [[float(len(p.quantiles)) - 1 for p in norm_params]]
        ).type(self.dtype)
        self._create_parameter(begin_index, "num_quantiles", num_quantiles)

        max_num_quantile_boundaries = int(
            torch.max(torch.tensor([len(p.quantiles) for p in norm_params]))
        )
        B = max_num_quantile_boundaries

        # The quantile boundaries is a FxB matrix where B is the max # of boundaries

        # We take advantage of the fact that if the value is >= the max
        # quantile boundary it automatically gets a 1.0 to repeat the max quantile
        # so that we guarantee a square matrix.

        # We project the quantiles boundaries to 3d and create a 1xFxB tensor
        quantile_boundaries = torch.zeros(
            [1, len(norm_params), max_num_quantile_boundaries]
        ).type(self.dtype)
        max_quantile_boundaries = torch.zeros([1, len(norm_params)]).type(self.dtype)
        min_quantile_boundaries = torch.zeros([1, len(norm_params)]).type(self.dtype)
        for i, p in enumerate(norm_params):
            quantile_boundaries[0, i, :] = p.quantiles[-1]
            quantile_boundaries[0, i, 0 : len(p.quantiles)] = torch.tensor(
                p.quantiles
            ).type(self.dtype)
            max_quantile_boundaries[0, i] = max(p.quantiles)
            min_quantile_boundaries[0, i] = min(p.quantiles)

        quantile_boundaries = quantile_boundaries.type(self.dtype)
        max_quantile_boundaries = max_quantile_boundaries.type(self.dtype)
        min_quantile_boundaries = min_quantile_boundaries.type(self.dtype)

        self._create_parameter(begin_index, "quantile_boundaries", quantile_boundaries)
        self._create_parameter(
            begin_index, "max_quantile_boundaries", max_quantile_boundaries
        )
        self._create_parameter(
            begin_index, "min_quantile_boundaries", min_quantile_boundaries
        )
        self._create_parameter(
            begin_index,
            "quantile_boundary_mask",
            torch.ones([1, F, B]).type(self.dtype),
        )

    def _preprocess_QUANTILE(
        self,
        begin_index: int,
        input: torch.Tensor,
        norm_params: List[NormalizationParameters],
    ) -> torch.Tensor:
        """
        Replace the value with it's percentile in the range [0,1].

        This preprocesses several features in a single step by putting the
        quantile boundaries in the third dimension and broadcasting.

        The input is a JxF matrix where J is the batch size and F is the # of features.
        """

        # The number of quantiles is a 1xF matrix
        num_quantiles = self._fetch_parameter(begin_index, "num_quantiles")

        quantile_boundaries = self._fetch_parameter(begin_index, "quantile_boundaries")
        max_quantile_boundaries = self._fetch_parameter(
            begin_index, "max_quantile_boundaries"
        )
        min_quantile_boundaries = self._fetch_parameter(
            begin_index, "min_quantile_boundaries"
        )

        # Add a third dimension and repeat to create a JxFxB matrix, where the
        # inputs are repeated B times in the third dimension.  We need to
        # do this because we can't broadcast both operands in different
        # dimensions in the same operation.

        # repeat doesn't work yet, so * by a mask
        mask = self._fetch_parameter(begin_index, "quantile_boundary_mask")
        expanded_inputs = input.unsqueeze(2) * mask

        input_greater_than_or_equal_to = (
            expanded_inputs >= quantile_boundaries
        ).float()

        input_less_than = (expanded_inputs < quantile_boundaries).float()
        set_to_max = (input >= max_quantile_boundaries).float()
        set_to_min = (input <= min_quantile_boundaries).float()
        min_or_max = (set_to_min + set_to_max).float()
        interpolate = (min_or_max < self.one_hundredth_tensor).float()
        interpolate_left, _ = torch.max(
            (input_greater_than_or_equal_to * quantile_boundaries)
            + (input_less_than * self.min_tensor),
            dim=2,
        )
        interpolate_right, _ = torch.min(
            (input_less_than * quantile_boundaries)
            + (input_greater_than_or_equal_to * self.max_tensor),
            dim=2,
        )

        # This assumes that we need to interpolate and computes the value.
        # If we don't need to interpolate, this will be some bogus value, but it
        # will be multiplied by 0 so no big deal.
        left_start = torch.sum(input_greater_than_or_equal_to, dim=2) - self.one_tensor
        interpolated_values = (
            (
                left_start
                + (
                    (input - interpolate_left)
                    / (
                        (interpolate_right + self.epsilon_tensor) - interpolate_left
                    )  # Add a small amount to interpolate_right to avoid div-0
                )
            )
            / num_quantiles
        ).float()
        return set_to_max + (interpolate * interpolated_values).float()

    def _create_parameters_ENUM(
        self, begin_index: int, norm_params: NormalizationParameters
    ):
        self._create_parameter(
            begin_index,
            "enum_values",
            torch.Tensor(norm_params.possible_values).unsqueeze(0).type(self.dtype),
        )

    def _preprocess_ENUM(
        self,
        begin_index: int,
        input: torch.Tensor,
        norm_params: NormalizationParameters,
    ) -> torch.Tensor:
        enum_values = self._fetch_parameter(begin_index, "enum_values")
        return (input == enum_values).float()

    def _sort_features_by_normalization(self):
        """
        Helper function to return a sorted list from a normalization map.
        Also returns the starting index for each feature type"""
        # Sort features by feature type
        sorted_features = []
        feature_starts = []
        assert isinstance(
            list(self.normalization_parameters.keys())[0], int
        ), "Normalization Parameters need to be int"
        for feature_type in FEATURE_TYPES:
            feature_starts.append(len(sorted_features))
            for feature in sorted(self.normalization_parameters.keys()):
                norm = self.normalization_parameters[feature]
                if norm.feature_type == feature_type:
                    sorted_features.append(feature)
        return sorted_features, feature_starts

    def _get_type_boundaries(self) -> List[int]:
        feature_starts = []
        on_feature_type = -1
        for i, feature in enumerate(self.sorted_features):
            feature_type = self.normalization_parameters[feature].feature_type
            feature_type_index = FEATURE_TYPES.index(feature_type)
            assert (
                feature_type_index >= on_feature_type
            ), "Features are not sorted by feature type!"
            while feature_type_index > on_feature_type:
                feature_starts.append(i)
                on_feature_type += 1
        while on_feature_type < len(FEATURE_TYPES):
            feature_starts.append(len(self.sorted_features))
            on_feature_type += 1
        return feature_starts

    def _create_parameter(
        self, begin_index: int, name: str, t: torch.Tensor
    ) -> Parameter:
        p = Parameter(t, requires_grad=False)
        setattr(self, "_auto_parameter_" + str(begin_index) + "_" + name, p)
        return p

    def _fetch_parameter(self, begin_index: int, name: str) -> Parameter:
        return getattr(self, "_auto_parameter_" + str(begin_index) + "_" + name)

    def _manual_broadcast_matrix_scalar(
        self, t1: torch.Tensor, s1: torch.Tensor, fn
    ) -> torch.Tensor:
        # Some ONNX ops don't support broadcasting so we need to do some matrix magic
        return fn(t1, (t1 * self.zero_tensor) + s1).float()

    def _manual_broadcast_column_vec_row_vec(
        self, t1: torch.Tensor, t2: torch.Tensor, fn
    ) -> torch.Tensor:
        # Some ONNX ops don't support broadcasting so we need to do some matrix magic
        t2_ones = t2 / t2
        t1_mask = t1.mm(t2_ones)

        return fn(t1_mask, t2).float()

    def _check_preprocessing_output(self, batch, norm_params):
        """
        Check that preprocessed features fall within range of valid output.
        :param batch: torch tensor
        :param norm_params: list of normalization parameters
        """
        feature_type = norm_params[0].feature_type
        min_value, max_value = batch.min(), batch.max()
        if feature_type == "CONTINUOUS":
            # Continuous features may be in range (-inf, inf)
            pass
        elif float(max_value) > MAX_FEATURE_VALUE:
            raise Exception(
                "A {} feature type has max value {} which is > than accepted post pre-processing max of {}".format(
                    feature_type, max_value, MAX_FEATURE_VALUE
                )
            )
        elif float(min_value) < MIN_FEATURE_VALUE:
            raise Exception(
                "A {} feature type has min value {} which is < accepted post pre-processing min of {}".format(
                    feature_type, min_value, MIN_FEATURE_VALUE
                )
            )


class PreprocesserAndForwardPassContainer(nn.Module):
    def __init__(self, preprocessor_module, forward_pass_module):
        super().__init__()
        self.add_module("preprocess", preprocessor_module)
        self.add_module("forward_pass", forward_pass_module)

    def forward(self, input) -> torch.FloatTensor:
        return self.forward_pass.forward(self.preprocess.forward(input))
