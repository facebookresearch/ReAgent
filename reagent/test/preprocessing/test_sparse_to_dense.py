#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import torch
from reagent.preprocessing import normalization
from reagent.preprocessing.sparse_to_dense import (
    PythonSparseToDenseProcessor,
    StringKeySparseToDenseProcessor,
)


class TestSparseToDense(unittest.TestCase):
    def setUp(self):
        self.sorted_features = [1, 2, 5, 4]
        self.str_keyed_sparse_data = [
            {},
            {"2": 0.3},
            {"4": 0.5, "5": 0.4},
            {"5": 0.3, "1": 0.5, "2": 0.1, "4": 0.7},
        ]
        self.int_keyed_sparse_data = [
            {int(k): v for k, v in d.items()} for d in self.str_keyed_sparse_data
        ]
        self.expected_value_0 = torch.tensor(
            [[0, 0, 0, 0], [0, 0.3, 0, 0], [0, 0, 0.4, 0.5], [0.5, 0.1, 0.3, 0.7]]
        )
        self.expected_presence_0 = torch.ones(4, 4).bool()
        MISSING = normalization.MISSING_VALUE
        self.expected_value_missing = torch.tensor(
            [
                [MISSING, MISSING, MISSING, MISSING],
                [MISSING, 0.3, MISSING, MISSING],
                [MISSING, MISSING, 0.4, 0.5],
                [0.5, 0.1, 0.3, 0.7],
            ]
        )
        self.expected_presence_missing = torch.tensor(
            [
                [False, False, False, False],
                [False, True, False, False],
                [False, False, True, True],
                [True, True, True, True],
            ]
        )

    def test_int_key_sparse_to_dense(self):
        # int keys, set_missing_value_to_zero=False
        processor = PythonSparseToDenseProcessor(
            self.sorted_features, set_missing_value_to_zero=False
        )
        value, presence = processor.process(self.int_keyed_sparse_data)
        assert torch.allclose(value, self.expected_value_missing)
        assert torch.all(presence == self.expected_presence_missing)

    def test_str_key_sparse_to_dense(self):
        # string keys, set_missing_value_to_zero=True
        processor = StringKeySparseToDenseProcessor(
            self.sorted_features, set_missing_value_to_zero=True
        )
        value, presence = processor.process(self.str_keyed_sparse_data)
        assert torch.allclose(value, self.expected_value_0)
        assert torch.all(presence == self.expected_presence_0)

        # string keys, set_missing_value_to_zero=False
        processor = StringKeySparseToDenseProcessor(
            self.sorted_features, set_missing_value_to_zero=False
        )
        value, presence = processor.process(self.str_keyed_sparse_data)
        assert torch.allclose(value, self.expected_value_missing)
        assert torch.all(presence == self.expected_presence_missing)
