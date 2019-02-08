#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from caffe2.python import workspace
from ml.rl.caffe_utils import C2, StackedAssociativeArray
from ml.rl.preprocessing import normalization
from ml.rl.preprocessing.normalization import MISSING_VALUE


logger = logging.getLogger(__name__)


class SparseToDenseProcessor:
    def process(
        self, sorted_features, sparse_data, set_missing_value_to_zero: bool = False
    ):
        raise NotImplementedError()

    def __call__(
        self, sorted_features, sparse_data, set_missing_value_to_zero: bool = False
    ):
        return self.process(sorted_features, sparse_data, set_missing_value_to_zero)


class Caffe2SparseToDenseProcessor(SparseToDenseProcessor):
    def process(
        self,
        sorted_features: List[int],
        sparse_data: StackedAssociativeArray,
        set_missing_value_to_zero: bool = False,
    ) -> Tuple[str, List[str]]:
        lengths_blob = sparse_data.lengths
        keys_blob = sparse_data.keys
        values_blob = sparse_data.values

        MISSING_SCALAR = C2.NextBlob("MISSING_SCALAR")
        missing_value = 0.0 if set_missing_value_to_zero else MISSING_VALUE
        workspace.FeedBlob(MISSING_SCALAR, np.array([missing_value], dtype=np.float32))
        C2.net().GivenTensorFill([], [MISSING_SCALAR], shape=[], values=[missing_value])

        parameters: List[str] = [MISSING_SCALAR]

        assert len(sorted_features) > 0, "Sorted features is empty"
        dense_input = C2.SparseToDenseMask(
            keys_blob, values_blob, MISSING_SCALAR, lengths_blob, mask=sorted_features
        )[0]

        return dense_input, parameters


class PandasSparseToDenseProcessor(SparseToDenseProcessor):
    def process(
        self, sorted_features, sparse_data, set_missing_value_to_zero: bool = False
    ) -> torch.Tensor:
        missing_value = normalization.MISSING_VALUE
        if set_missing_value_to_zero:
            missing_value = 0.0
        state_features_df = pd.DataFrame(sparse_data).fillna(missing_value)
        # Add columns identified by normalization, but not present in batch
        for col in sorted_features:
            if col not in state_features_df.columns:
                state_features_df[col] = missing_value
        return torch.from_numpy(state_features_df[sorted_features].values).float()
