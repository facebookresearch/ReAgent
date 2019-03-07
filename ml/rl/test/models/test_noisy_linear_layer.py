#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import logging
import unittest

import torch
from ml.rl.caffe_utils import C2, PytorchCaffe2Converter
from ml.rl.models.noisy_linear_layer import NoisyLinear
from ml.rl.test.models.test_utils import check_save_load


logger = logging.getLogger(__name__)


class TestNoisyLinearLayer(unittest.TestCase):
    def test_non_determinism(self):
        in_dim, out_dim = 5, 1
        noisy_layer = NoisyLinear(in_dim, out_dim)
        input = torch.randn(1, 5)
        equal_tensor = torch.eq(noisy_layer(input), noisy_layer(input))
        self.assertEqual(equal_tensor.sum(), 0)
