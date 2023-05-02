#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import torch
from reagent.models.cb_fully_connected_network import CBFullyConnectedNetwork


class TestCBFullyConnectedNetwork(unittest.TestCase):
    def test_call_no_ucb(self) -> None:
        model = CBFullyConnectedNetwork(2, [5, 7], activation="relu")

        inp = torch.tensor([[1.0, 5.0], [1.0, 6.0]])
        model_output = model(inp)
        pred_label = model_output["pred_label"]
        ucb = model_output["ucb"]

        self.assertIsInstance(pred_label, torch.Tensor)
        self.assertEqual(tuple(pred_label.shape), (2,))
        assert torch.allclose(pred_label, ucb, atol=1e-4, rtol=1e-4)
