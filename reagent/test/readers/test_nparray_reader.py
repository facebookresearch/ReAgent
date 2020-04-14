#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest
from collections import OrderedDict

import numpy as np
import numpy.testing as npt
from ml.rl.readers.nparray_reader import NpArrayReader


class TestNpArrayReader(unittest.TestCase):
    def get_test_data(self, n):
        return OrderedDict(
            [
                ("states", np.random.randn(n, 10)),
                ("actions", np.random.randn(n, 10)),
                ("rewards", np.random.randn(n)),
                (
                    "next",
                    OrderedDict(
                        [
                            ("states", np.random.randn(n, 10)),
                            ("actions", np.random.randn(n, 10)),
                        ]
                    ),
                ),
            ]
        )

    def assert_batch_equal(self, data, batch, offset, length):
        for k in ["states", "actions", "rewards"]:
            npt.assert_array_equal(data[k][offset : offset + length], batch[k])
        for k in ["states", "actions"]:
            npt.assert_array_equal(
                data["next"][k][offset : offset + length], batch["next"][k]
            )

    def test_basic(self):
        data = self.get_test_data(1000)
        batch_size = 100
        reader = NpArrayReader(data, batch_size=batch_size)
        for i, batch in enumerate(reader):
            self.assert_batch_equal(data, batch, i * batch_size, batch_size)
        self.assertEqual(9, i)

    def test_drop_small(self):
        data = self.get_test_data(999)
        batch_size = 100
        reader = NpArrayReader(data, batch_size=batch_size)
        for i, batch in enumerate(reader):
            self.assert_batch_equal(data, batch, i * batch_size, batch_size)
        self.assertEqual(8, i)

    def test_not_drop_small(self):
        data = self.get_test_data(999)
        batch_size = 100
        reader = NpArrayReader(data, batch_size=batch_size, drop_small=False)
        for i, batch in enumerate(reader):
            self.assert_batch_equal(
                data, batch, i * batch_size, batch_size if i != 9 else 99
            )
        self.assertEqual(9, i)

    def test_shard(self):
        n = 1000
        data = self.get_test_data(n)
        batch_size = 100
        num_shards = 2
        shard_size = n // num_shards
        reader = NpArrayReader(data, batch_size=batch_size, num_shards=num_shards)
        num_batches = 0
        for shard in range(num_shards):
            for i, batch in enumerate(reader.get_shard(shard)):
                self.assert_batch_equal(
                    data, batch, shard * shard_size + i * batch_size, batch_size
                )
                num_batches += 1
        self.assertEqual(10, num_batches)

    def test_shard_drop_small(self):
        n = 1000
        data = self.get_test_data(n)
        batch_size = 100
        num_shards = 3
        shard_size = n // num_shards + 1
        reader = NpArrayReader(data, batch_size=batch_size, num_shards=num_shards)
        num_batches = 0
        for shard in range(num_shards):
            for i, batch in enumerate(reader.get_shard(shard)):
                self.assert_batch_equal(
                    data, batch, shard * shard_size + i * batch_size, batch_size
                )
                num_batches += 1
        self.assertEqual(9, num_batches)

    def test_shard_not_drop_small(self):
        n = 1000
        data = self.get_test_data(n)
        batch_size = 100
        num_shards = 3
        shard_size = n // num_shards + 1
        reader = NpArrayReader(
            data, batch_size=batch_size, drop_small=False, num_shards=num_shards
        )
        num_batches = 0
        for shard in range(num_shards):
            for i, batch in enumerate(reader.get_shard(shard)):
                self.assert_batch_equal(
                    data,
                    batch,
                    shard * shard_size + i * batch_size,
                    batch_size if i != 3 else (34 if shard != 2 else 32),
                )
                num_batches += 1
        self.assertEqual(12, num_batches)
