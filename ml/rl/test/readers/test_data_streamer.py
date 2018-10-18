#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest
from collections import OrderedDict

import numpy as np
import numpy.testing as npt
from ml.rl.readers.data_streamer import DataStreamer
from ml.rl.readers.nparray_reader import NpArrayReader


class TestDataStreamer(unittest.TestCase):
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
        streamer = DataStreamer(reader)
        for i, batch in enumerate(streamer):
            self.assert_batch_equal(data, batch, i * batch_size, batch_size)
        self.assertEqual(9, i)

    def test_drop_small(self):
        data = self.get_test_data(999)
        batch_size = 100
        reader = NpArrayReader(data, batch_size=batch_size)
        streamer = DataStreamer(reader)
        for i, batch in enumerate(streamer):
            self.assert_batch_equal(data, batch, i * batch_size, batch_size)
        self.assertEqual(8, i)

    def test_not_drop_small(self):
        data = self.get_test_data(999)
        batch_size = 100
        reader = NpArrayReader(data, batch_size=batch_size, drop_small=False)
        streamer = DataStreamer(reader)
        for i, batch in enumerate(streamer):
            self.assert_batch_equal(
                data, batch, i * batch_size, batch_size if i != 9 else 99
            )
        self.assertEqual(9, i)

    def test_basic_one_worker(self):
        data = self.get_test_data(1000)
        batch_size = 100
        reader = NpArrayReader(data, batch_size=batch_size, num_shards=1)
        streamer = DataStreamer(reader, num_workers=1)
        for i, batch in enumerate(streamer):
            self.assert_batch_equal(data, batch, i * batch_size, batch_size)
        self.assertEqual(9, i)

    def test_drop_small_one_worker(self):
        data = self.get_test_data(999)
        batch_size = 100
        reader = NpArrayReader(data, batch_size=batch_size, num_shards=1)
        streamer = DataStreamer(reader, num_workers=1)
        for i, batch in enumerate(streamer):
            self.assert_batch_equal(data, batch, i * batch_size, batch_size)
        self.assertEqual(8, i)

    def test_not_drop_small_one_worker(self):
        data = self.get_test_data(999)
        batch_size = 100
        reader = NpArrayReader(
            data, batch_size=batch_size, drop_small=False, num_shards=1
        )
        streamer = DataStreamer(reader, num_workers=1)
        for i, batch in enumerate(streamer):
            self.assert_batch_equal(
                data, batch, i * batch_size, batch_size if i != 9 else 99
            )
        self.assertEqual(9, i)

    def test_basic_two_workers(self):
        data = self.get_test_data(1000)
        batch_size = 100
        num_shards = 2
        reader = NpArrayReader(data, batch_size=batch_size, num_shards=num_shards)

        splits = [reader._get_split(reader.data, i, batch_size) for i in range(10)]

        streamer = DataStreamer(reader, num_workers=num_shards)
        for _i, batch in enumerate(streamer):
            match = False
            for split in splits:
                try:
                    self.assert_batch_equal(split, batch, 0, batch_size)
                except Exception:
                    pass
                else:
                    match = True
                    break
            self.assertTrue(match)
        self.assertEqual(9, _i)
