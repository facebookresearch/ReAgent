#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import dataclasses
import logging
import os
import tempfile
import unittest
from io import BytesIO
from typing import Any, NamedTuple

import numpy as np
import numpy.testing as npt
import torch
import torch.nn as nn
from reagent.models.base import ModelBase
from reagent.preprocessing.identify_types import CONTINUOUS
from reagent.preprocessing.normalization import NormalizationParameters
from reagent.test.models.test_utils import check_save_load
from reagent.types import FeatureVector, PreprocessedStateAction


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelOutput:
    # These should be torch.Tensor but the type checking failed when I used it
    sum: Any
    mul: Any
    plus_one: Any
    linear: Any


class Model(ModelBase):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def input_prototype(self):
        return PreprocessedStateAction.from_tensors(
            state=torch.randn([1, 4]), action=torch.randn([1, 4])
        )

    def forward(self, sa):
        state = sa.state.float_features
        action = sa.action.float_features

        return ModelOutput(
            state + action, state * action, state + 1, self.linear(state)
        )


class TestBase(unittest.TestCase):
    def test_get_predictor_export_meta_and_workspace(self):
        model = Model()

        # 2 params + 1 const
        expected_num_params, expected_num_inputs, expected_num_outputs = 3, 2, 4
        check_save_load(
            self, model, expected_num_params, expected_num_inputs, expected_num_outputs
        )
