#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import torch
from reagent.models.cb_fully_connected_network import CBFullyConnectedNetwork


class TestCBFullyConnectedNetwork(unittest.TestCase):
    def test_call_no_ucb(self) -> None:
        model = CBFullyConnectedNetwork(2, [5, 7], activation="relu")

        inp = torch.tensor([[1.0, 5.0], [1.0, 6.0]])
        out = model(inp)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(tuple(out.shape), (2,))
