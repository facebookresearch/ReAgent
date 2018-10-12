#!/usr/bin/env python3


import logging
from typing import List, Tuple

import numpy as np
from caffe2.python import workspace
from ml.rl.caffe_utils import C2
from ml.rl.preprocessing.normalization import MISSING_VALUE


logger = logging.getLogger(__name__)


def sparse_to_dense(
    lengths_blob: str, keys_blob: str, values_blob: str, sorted_features: List[int]
) -> Tuple[str, List[str]]:
    MISSING_SCALAR = C2.NextBlob("MISSING_SCALAR")
    workspace.FeedBlob(MISSING_SCALAR, np.array([MISSING_VALUE], dtype=np.float32))
    C2.net().GivenTensorFill([], [MISSING_SCALAR], shape=[], values=[MISSING_VALUE])

    parameters: List[str] = [MISSING_SCALAR]

    assert len(sorted_features) > 0, "Sorted features is empty"
    dense_input = C2.SparseToDenseMask(
        keys_blob, values_blob, MISSING_SCALAR, lengths_blob, mask=sorted_features
    )[0]

    return dense_input, parameters
