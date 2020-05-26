#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
from typing import Dict, List, Optional, Tuple, cast

import torch
from reagent.parameters import NormalizationParameters
from reagent.preprocessing.identify_types import ENUM, FEATURE_TYPES
from reagent.preprocessing.normalization import (
    EPS,
    MAX_FEATURE_VALUE,
    MIN_FEATURE_VALUE,
)
from torch.nn import Module, Parameter


logger = logging.getLogger(__name__)


class Preprocessor(Module):
    def __init__(
        self,
        normalization_parameters: Dict[int, NormalizationParameters],
        use_gpu: Optional[bool] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.normalization_parameters = normalization_parameters
        (
            self.feature_id_to_index,
            self.sorted_features,
            self.sorted_feature_boundaries,
        ) = self._sort_features_by_normalization()

        cuda_available = torch.cuda.is_available()
        logger.info("CUDA availability: {}".format(cuda_available))
        if device is not None:
            self.device = device
        elif use_gpu and cuda_available:
            logger.warn("use_gpu is deprecated, please pass in the device directly")
            logger.info("Using GPU: GPU requested and available.")
            self.device = torch.device("cuda")
        else:
            logger.info("NOT Using GPU: GPU not requested or not available.")
            self.device = torch.device("cpu")

        # NOTE: Because of the way we call AppendNet to squash ONNX to a C2 net,
        # We need to make tensors for every numeric literal
        self.zero_tensor = Parameter(
            torch.tensor([0.0], device=self.device), requires_grad=False
        )
        self.one_tensor = Parameter(
            torch.tensor([1.0], device=self.device), requires_grad=False
        )
        self.one_half_tensor = Parameter(
            torch.tensor([0.5], device=self.device), requires_grad=False
        )
        self.one_hundredth_tensor = Parameter(
            torch.tensor([0.01], device=self.device), requires_grad=False
        )
        self.negative_one_tensor = Parameter(
            torch.tensor([-1.0], device=self.device), requires_grad=False
        )
        self.min_tensor = Parameter(
            torch.tensor([-1e20], device=self.device), requires_grad=False
        )
        self.max_tensor = Parameter(
            torch.tensor([1e20], device=self.device), requires_grad=False
        )
        self.epsilon_tensor = Parameter(
            torch.tensor([EPS], device=self.device), requires_grad=False
        )

        self.feature_starts = self._get_type_boundaries()
        self.split_sections: List[int] = []
        for i, feature_type in enumerate(FEATURE_TYPES):
            begin_index = self.feature_starts[i]
            if (i + 1) == len(FEATURE_TYPES):
                end_index = len(self.normalization_parameters)
            else:
                end_index = self.feature_starts[i + 1]
            if begin_index == end_index:
                continue  # No features of this type
            if feature_type == ENUM:
                # Process one-at-a-time
                for j in range(begin_index, end_index):
                    enum_norm_params = self.normalization_parameters[
                        self.sorted_features[j]
                    ]
                    func = getattr(self, "_create_parameters_" + feature_type)
                    func(j, enum_norm_params)
                    self.split_sections.append(1)
            else:
                norm_params = []
                for f in self.sorted_features[begin_index:end_index]:
                    norm_params.append(self.normalization_parameters[f])
                func = getattr(self, "_create_parameters_" + feature_type)
                func(begin_index, norm_params)
                self.split_sections.append(end_index - begin_index)

    def input_prototype(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.randn(1, len(self.normalization_parameters), device=self.device),
            torch.ones(
                1,
                len(self.normalization_parameters),
                dtype=torch.uint8,
                device=self.device,
            ),
        )

    def forward(
        self, input: torch.Tensor, input_presence_byte: torch.Tensor
    ) -> torch.Tensor:
        """ Preprocess the input matrix
        :param input tensor
        """
        outputs = []
        split_input = torch.split(input, self.split_sections, dim=1)
        # NB: converting to float prevent ASAN heap-buffer-overflow
        split_presence = torch.split(
            input_presence_byte.float(), self.split_sections, dim=1
        )
        ptr = 0
        for i, feature_type in enumerate(FEATURE_TYPES):
            begin_index = self.feature_starts[i]
            if (i + 1) == len(FEATURE_TYPES):
                end_index = len(self.normalization_parameters)
            else:
                end_index = self.feature_starts[i + 1]
            if begin_index == end_index:
                continue  # No features of this type
            if feature_type == ENUM:
                # Process one-at-a-time
                for j in range(begin_index, end_index):
                    norm_params = self.normalization_parameters[self.sorted_features[j]]
                    new_output = (
                        self._preprocess_feature_single_column(
                            j, split_input[ptr], norm_params
                        )
                        * split_presence[ptr]
                    )
                    ptr += 1
                    self._check_preprocessing_output(new_output, [norm_params])
                    outputs.append(new_output)
            else:
                norm_params_list: List[NormalizationParameters] = []
                for f in self.sorted_features[begin_index:end_index]:
                    norm_params_list.append(self.normalization_parameters[f])
                new_output = (
                    self._preprocess_feature_multi_column(
                        begin_index, split_input[ptr], norm_params_list
                    )
                    * split_presence[ptr]
                )
                ptr += 1
                self._check_preprocessing_output(new_output, norm_params_list)
                outputs.append(new_output)

        if len(outputs) == 1:
            return cast(
                torch.Tensor,
                torch.clamp(outputs[0], MIN_FEATURE_VALUE, MAX_FEATURE_VALUE),
            )

        return cast(
            torch.Tensor,
            torch.clamp(
                torch.cat(outputs, dim=1), MIN_FEATURE_VALUE, MAX_FEATURE_VALUE
            ),
        )

    def _preprocess_feature_single_column(
        self,
        begin_index: int,
        input: torch.Tensor,
        norm_params: NormalizationParameters,
    ) -> torch.Tensor:
        feature_type = norm_params.feature_type
        func = getattr(self, "_preprocess_" + feature_type)
        return func(begin_index, input, norm_params)

    def _preprocess_feature_multi_column(
        self,
        begin_index: int,
        input: torch.Tensor,
        norm_params: List[NormalizationParameters],
    ) -> torch.Tensor:
        feature_type = norm_params[0].feature_type
        func = getattr(self, "_preprocess_" + feature_type)
        return func(begin_index, input, norm_params)

    def _create_parameters_DO_NOT_PREPROCESS(
        self, begin_index: int, norm_params: List[NormalizationParameters]
    ):
        pass

    def _preprocess_DO_NOT_PREPROCESS(
        self,
        begin_index: int,
        input: torch.Tensor,
        norm_params: List[NormalizationParameters],
    ) -> torch.Tensor:
        return input

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
            torch.tensor([p.min_value for p in norm_params], device=self.device),
        )
        self._create_parameter(
            begin_index,
            "min_training_value",
            torch.ones(len(norm_params), device=self.device) * -1 + EPS,
        )
        self._create_parameter(
            begin_index,
            "scaling_factor",
            (torch.ones(len(norm_params), device=self.device) - EPS)
            * 2
            / torch.tensor(
                # pyre-fixme[16]: `Optional` has no attribute `__sub__`.
                [p.max_value - p.min_value for p in norm_params],
                device=self.device,
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
            torch.tensor([p.mean for p in norm_params], device=self.device),
        )
        self._create_parameter(
            begin_index,
            "stddevs",
            torch.tensor([p.stddev for p in norm_params], device=self.device),
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
        return continuous_output

    def _create_parameters_BOXCOX(
        self, begin_index: int, norm_params: List[NormalizationParameters]
    ):
        self._create_parameter(
            begin_index,
            "shifts",
            torch.tensor([p.boxcox_shift for p in norm_params], device=self.device),
        )
        for p in norm_params:
            assert (
                # pyre-fixme[16]: `Optional` has no attribute `__abs__`.
                abs(p.boxcox_lambda)
                > 1e-6
            ), "Invalid value for boxcox lambda: " + str(p.boxcox_lambda)
        self._create_parameter(
            begin_index,
            "lambdas",
            torch.tensor([p.boxcox_lambda for p in norm_params], device=self.device),
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
            torch.pow(
                torch.clamp(
                    input + shifts, 1e-6
                ),  # Clamp is necessary to prevent MISSING_VALUE from going to NaN
                lambdas,
            )
            - self.one_tensor
        ) / lambdas
        return self._preprocess_CONTINUOUS(begin_index, boxcox_output, norm_params)

    def _create_parameters_QUANTILE(
        self, begin_index: int, norm_params: List[NormalizationParameters]
    ):
        F = len(norm_params)

        num_quantiles = torch.tensor(
            # pyre-fixme[6]: Expected `Sized` for 1st param but got
            #  `Optional[List[float]]`.
            [[float(len(p.quantiles)) - 1 for p in norm_params]],
            device=self.device,
        )
        self._create_parameter(begin_index, "num_quantiles", num_quantiles)

        max_num_quantile_boundaries = int(
            torch.max(
                # pyre-fixme[6]: Expected `Sized` for 1st param but got
                #  `Optional[List[float]]`.
                torch.tensor([len(p.quantiles) for p in norm_params])
            ).item()
        )
        B = max_num_quantile_boundaries

        # The quantile boundaries is a FxB matrix where B is the max # of boundaries

        # We take advantage of the fact that if the value is >= the max
        # quantile boundary it automatically gets a 1.0 to repeat the max quantile
        # so that we guarantee a square matrix.

        # We project the quantiles boundaries to 3d and create a 1xFxB tensor
        quantile_boundaries = torch.zeros(
            [1, len(norm_params), max_num_quantile_boundaries], device=self.device
        )
        max_quantile_boundaries = torch.zeros([1, len(norm_params)], device=self.device)
        min_quantile_boundaries = torch.zeros([1, len(norm_params)], device=self.device)
        for i, p in enumerate(norm_params):
            # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
            quantile_boundaries[0, i, :] = p.quantiles[-1]
            quantile_boundaries[
                0,
                i,
                # pyre-fixme[6]: Expected `Sized` for 1st param but got
                #  `Optional[List[float]]`.
                0 : len(p.quantiles),
            ] = torch.tensor(p.quantiles, device=self.device)
            # pyre-fixme[6]: Expected `Iterable[Variable[_T]]` for 1st param but got
            #  `Optional[List[float]]`.
            max_quantile_boundaries[0, i] = max(p.quantiles)
            # pyre-fixme[6]: Expected `Iterable[Variable[_T]]` for 1st param but got
            #  `Optional[List[float]]`.
            min_quantile_boundaries[0, i] = min(p.quantiles)

        quantile_boundaries = quantile_boundaries.to(self.device)
        max_quantile_boundaries = max_quantile_boundaries.to(self.device)
        min_quantile_boundaries = min_quantile_boundaries.to(self.device)

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
            torch.ones([1, F, B], device=self.device),
        )

    def _preprocess_QUANTILE(
        self,
        begin_index: int,
        input: torch.Tensor,
        norm_params: List[NormalizationParameters],
    ) -> torch.Tensor:
        """
        Replace the value with its percentile in the range [0,1].

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
            torch.tensor(norm_params.possible_values, device=self.device).unsqueeze(0),
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
        feature_id_to_index = {}
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
                    feature_id_to_index[feature] = len(sorted_features)
                    sorted_features.append(feature)
        return feature_id_to_index, sorted_features, feature_starts

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
        return cast(
            Parameter, getattr(self, "_auto_parameter_" + str(begin_index) + "_" + name)
        )

    def _check_preprocessing_output(self, batch, norm_params):
        """
        Check that preprocessed features fall within range of valid output.
        :param batch: torch tensor
        :param norm_params: list of normalization parameters
        """
        if not self.training:
            return
        feature_type = norm_params[0].feature_type
        min_value, max_value = batch.min(), batch.max()

        if feature_type in ("BOXCOX", "CONTINUOUS"):
            # Continuous features may be in range (-inf, inf)
            pass
        elif max_value.item() > MAX_FEATURE_VALUE:
            raise Exception(
                f"A {feature_type} feature type has max value {max_value} which is >"
                f" than accepted post pre-processing max of {MAX_FEATURE_VALUE}"
            )
        elif min_value.item() < MIN_FEATURE_VALUE:
            raise Exception(
                f"A {feature_type} feature type has min value {min_value} which is <"
                f" accepted post pre-processing min of {MIN_FEATURE_VALUE}"
            )
