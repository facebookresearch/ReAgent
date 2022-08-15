#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import reagent.core.types as rlt
import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TestTypes(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_tensor_data_class_to_cuda(self):
        # Test if TensorDataClass.to(...) can move to designated device
        batch_size = 4
        dim = 5
        float_features = torch.randn(batch_size, dim)
        keys = ["Key0", "Key1", "Key2"]
        values = torch.arange(10).float()
        lengths = torch.tensor([2, 0, 1, 1, 1, 1, 3, 0, 0, 1, 0, 0])
        id_list_features = KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)
        id_list_features_raw = {
            "key0": (torch.randn(batch_size, dim), torch.randn(batch_size, dim))
        }
        data = rlt.FeatureData(
            float_features=float_features,
            id_list_features=id_list_features,
            id_list_features_raw=id_list_features_raw,
        )
        data_cuda = data.to(torch.device("cuda"))
        assert data_cuda.float_features.device.type == "cuda"
        assert data_cuda.id_list_features.values().device.type == "cuda"
        assert data_cuda.id_list_features.lengths().device.type == "cuda"
        assert data_cuda.id_list_features_raw["key0"][0].device.type == "cuda"
        assert data_cuda.id_list_features_raw["key0"][1].device.type == "cuda"
