#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import logging
from typing import Dict, List, Tuple

import numpy as np

# @manual=third-party//pandas:pandas-py
import pandas as pd
import torch
from caffe2.python import workspace
from ml.rl.caffe_utils import C2, StackedAssociativeArray
from ml.rl.preprocessing import normalization
from ml.rl.preprocessing.normalization import MISSING_VALUE


logger = logging.getLogger(__name__)


class SparseToDenseProcessor:
    def __init__(
        self, sorted_features: List[int], set_missing_value_to_zero: bool = False
    ):
        self.sorted_features = sorted_features
        self.set_missing_value_to_zero = set_missing_value_to_zero

    def __call__(self, sparse_data):
        return self.process(sparse_data)


class Caffe2SparseToDenseProcessor(SparseToDenseProcessor):
    def __init__(
        self, sorted_features: List[int], set_missing_value_to_zero: bool = False
    ):
        super().__init__(sorted_features, set_missing_value_to_zero)

    def process(
        self, sparse_data: StackedAssociativeArray
    ) -> Tuple[str, str, List[str]]:
        lengths_blob = sparse_data.lengths
        keys_blob = sparse_data.keys
        values_blob = sparse_data.values

        MISSING_SCALAR = C2.NextBlob("MISSING_SCALAR")
        missing_value = 0.0 if self.set_missing_value_to_zero else MISSING_VALUE
        workspace.FeedBlob(MISSING_SCALAR, np.array([missing_value], dtype=np.float32))
        C2.net().GivenTensorFill([], [MISSING_SCALAR], shape=[], values=[missing_value])

        parameters: List[str] = [MISSING_SCALAR]

        assert len(self.sorted_features) > 0, "Sorted features is empty"
        dense_input = C2.NextBlob("dense_input")
        dense_input_presence = C2.NextBlob("dense_input_presence")
        C2.net().SparseToDenseMask(
            [keys_blob, values_blob, MISSING_SCALAR, lengths_blob],
            [dense_input, dense_input_presence],
            mask=self.sorted_features,
            return_presence_mask=True,
        )

        if self.set_missing_value_to_zero:
            dense_input_presence = C2.And(
                C2.GT(dense_input, -1e-4, broadcast=1),
                C2.LT(dense_input, 1e-4, broadcast=1),
            )

        return dense_input, dense_input_presence, parameters


class PandasSparseToDenseProcessor(SparseToDenseProcessor):
    def __init__(
        self, sorted_features: List[int], set_missing_value_to_zero: bool = False
    ):
        super().__init__(sorted_features, set_missing_value_to_zero)

    def process(self, sparse_data) -> Tuple[torch.Tensor, torch.Tensor]:
        missing_value = normalization.MISSING_VALUE
        if self.set_missing_value_to_zero:
            missing_value = 0.0
        state_features_df = pd.DataFrame(sparse_data).fillna(missing_value)
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
