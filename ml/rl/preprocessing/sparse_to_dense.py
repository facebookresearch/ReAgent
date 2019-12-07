#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import logging
from typing import Dict, List, Tuple

# @manual=third-party//pandas:pandas-py
import pandas as pd
import torch
from ml.rl.preprocessing import normalization


logger = logging.getLogger(__name__)


class SparseToDenseProcessor:
    def __init__(
        self, sorted_features: List[int], set_missing_value_to_zero: bool = False
    ):
        self.sorted_features = sorted_features
        self.set_missing_value_to_zero = set_missing_value_to_zero

    def __call__(self, sparse_data):
        return self.process(sparse_data)


class PandasSparseToDenseProcessor(SparseToDenseProcessor):
    def __init__(
        self, sorted_features: List[int], set_missing_value_to_zero: bool = False
    ):
        super().__init__(sorted_features, set_missing_value_to_zero)
        self.feature_to_index: Dict[int, int] = {}
        for i, f in enumerate(sorted_features):
            self.feature_to_index[f] = i

    def process(self, sparse_data) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert all keys to integers
        sparse_data_int = []
        for sd in sparse_data:
            sd_int = {}
            for k, v in sd.items():
                sd_int[self.feature_to_index[int(k)]] = v
            sparse_data_int.append(sd_int)
        missing_value = normalization.MISSING_VALUE
        if self.set_missing_value_to_zero:
            missing_value = 0.0
        state_features_df = pd.DataFrame(sparse_data_int).fillna(missing_value)
        # Add columns identified by normalization, but not present in batch
        for col in self.sorted_features:
            if col not in state_features_df.columns:
                state_features_df[col] = missing_value
        values = torch.from_numpy(
            state_features_df[self.sorted_features].values
        ).float()
        if self.set_missing_value_to_zero:
            # When we set missing values to 0, we don't know what is and isn't missing
            presence = values != 0.0
        else:
            presence = values != missing_value
        return values, presence


class PythonSparseToDenseProcessor(SparseToDenseProcessor):
    def __init__(
        self, sorted_features: List[int], set_missing_value_to_zero: bool = False
    ):
        super().__init__(sorted_features, set_missing_value_to_zero)
        self.feature_id_to_index: Dict[int, int] = {}
        for index, feature_id in enumerate(sorted_features):
            self.feature_id_to_index[feature_id] = index

    def process(
        self, sparse_data: List[Dict[int, float]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dense_data = torch.ones([len(sparse_data), len(self.feature_id_to_index)])
        dense_presence = torch.zeros(
            [len(sparse_data), len(self.feature_id_to_index)]
        ).byte()
        for i, feature_map in enumerate(sparse_data):
            assert (
                feature_map is not None
            ), f"Please make sure that features are not NULL; row {i}"
            for j, value in feature_map.items():
                j_index = self.feature_id_to_index.get(j, None)
                if j_index is None:
                    continue
                dense_data[i][j_index] = value
                dense_presence[i][j_index] = 1
        if self.set_missing_value_to_zero:
            # When we set missing values to 0, we don't know what is and isn't missing
            dense_presence = dense_data != 0.0
        return (dense_data, dense_presence)
