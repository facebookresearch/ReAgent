#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import Dict, List, Tuple

import torch
from reagent.preprocessing import normalization


class SparseToDenseProcessor:
    def __init__(
        self, sorted_features: List[int], set_missing_value_to_zero: bool = False
    ) -> None:
        self.sorted_features = sorted_features
        self.set_missing_value_to_zero = set_missing_value_to_zero

    def __call__(self, sparse_data):
        return self.process(sparse_data)


class StringKeySparseToDenseProcessor(SparseToDenseProcessor):
    """
    We just have this in case the input data is keyed by string
    """

    def __init__(
        self, sorted_features: List[int], set_missing_value_to_zero: bool = False
    ) -> None:
        super().__init__(sorted_features, set_missing_value_to_zero)
        self._sparse_to_dense = PythonSparseToDenseProcessor(
            sorted_features, set_missing_value_to_zero
        )

    def process(
        self, sparse_data: List[Dict[str, float]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert all keys to integers
        sparse_data_int = []
        for sd in sparse_data:
            sd_int = {}
            for k, v in sd.items():
                sd_int[int(k)] = v
            sparse_data_int.append(sd_int)
        return self._sparse_to_dense(sparse_data_int)


class PythonSparseToDenseProcessor(SparseToDenseProcessor):
    def __init__(
        self, sorted_features: List[int], set_missing_value_to_zero: bool = False
    ) -> None:
        super().__init__(sorted_features, set_missing_value_to_zero)
        self.feature_to_index: Dict[int, int] = {
            f: i for i, f in enumerate(sorted_features)
        }

    def process(
        self, sparse_data: List[Dict[int, float]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        missing_value = normalization.MISSING_VALUE
        if self.set_missing_value_to_zero:
            missing_value = 0.0
        values = torch.nan_to_num(
            torch.FloatTensor(
                [
                    [
                        row[col] if col in row else missing_value
                        for col in self.sorted_features
                    ]
                    for row in sparse_data
                ]
            ),
            nan=missing_value,
        )
        if self.set_missing_value_to_zero:
            # When we set missing values to 0, we don't know what is and isn't missing
            presence = torch.ones_like(values, dtype=torch.bool)
        else:
            presence = values != missing_value
        return values, presence
