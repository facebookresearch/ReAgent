#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import numpy.testing as npt
import torch
from reagent.preprocessing.identify_types import CONTINUOUS_ACTION, DO_NOT_PREPROCESS
from reagent.preprocessing.normalization import (
    NormalizationData,
    NormalizationParameters,
)
from reagent.preprocessing.postprocessor import Postprocessor
from reagent.preprocessing.preprocessor import Preprocessor


class TestPostprocessing(unittest.TestCase):
    def test_continuous_action(self):
        normalization_params = {
            i: NormalizationParameters(
                feature_type=CONTINUOUS_ACTION, min_value=-5.0 * i, max_value=10.0 * i
            )
            for i in range(1, 5)
        }
        preprocessor = Preprocessor(normalization_params, use_gpu=False)
        postprocessor = Postprocessor(normalization_params, use_gpu=False)

        x = torch.rand(3, 4) * torch.tensor([15, 30, 45, 60]) + torch.tensor(
            [-5, -10, -15, -20]
        )
        presence = torch.ones_like(x, dtype=torch.uint8)
        y = postprocessor(preprocessor(x, presence))
        npt.assert_allclose(x, y, rtol=1e-4)

    def test_do_not_preprocess(self):
        normalization_params = {
            i: NormalizationParameters(feature_type=DO_NOT_PREPROCESS)
            for i in range(1, 5)
        }
        preprocessor = Preprocessor(normalization_params, use_gpu=False)
        postprocessor = Postprocessor(normalization_params, use_gpu=False)

        x = torch.randn(3, 4)
        presence = torch.ones_like(x, dtype=torch.uint8)
        y = postprocessor(preprocessor(x, presence))
        npt.assert_allclose(x, y)
