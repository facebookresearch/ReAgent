#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

from ml.rl.models.example_sequence_model import ExampleSequenceModel
from ml.rl.test.models.test_utils import check_save_load


logger = logging.getLogger(__name__)


class TestExampleSequenceModel(unittest.TestCase):
    def test_basic(self):
        state_dim = 8
        model = ExampleSequenceModel(state_dim)
        input = model.input_prototype()
        output = model(input)
        self.assertEqual((1, 1), output.value.shape)

    def test_save_load(self):
        state_dim = 8
        model = ExampleSequenceModel(state_dim)
        # ONNX sure exports a lot of parameters...
        expected_num_params, expected_num_inputs, expected_num_outputs = 133, 3, 1
        check_save_load(
            self, model, expected_num_params, expected_num_inputs, expected_num_outputs
        )
