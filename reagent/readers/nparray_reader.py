#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math
from collections import OrderedDict
from typing import Optional

import numpy as np

from .base import ReaderBase, ReaderIter


class NpArrayReaderIter(ReaderIter):
    def __init__(self, reader):
        self.reader = reader
        self.batch_read = 0
        self.total_batches = (
            reader.size // reader.batch_size
            if reader.drop_small
            else int(math.ceil(reader.size / reader.batch_size))
        )

    def read_batch(self) -> Optional[OrderedDict]:
        if self.batch_read == self.total_batches:
            return None
        ret = self.reader._get_split(
            self.reader.data, self.batch_read, self.reader.batch_size
        )
        self.batch_read += 1
        return ret


class NpArrayReader(ReaderBase):
    """
    Basic reader taking `np.ndarray`s of a whole dataset and split them into
    chunks of `batch_size`.
    """

    def __init__(self, data, size=None, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.size = size
        self._sanity_check_data(self.data)

    def _sanity_check_data(self, data):
        if isinstance(data, OrderedDict):
            for v in data.values():
                self._sanity_check_data(v)
        elif isinstance(data, np.ndarray):
            if self.size is None:
                self.size = data.shape[0]
            assert data.shape[0] == self.size
        else:
            raise ValueError("Got unexpected type {}".format(type(data)))

    def do_get_shard(self, shard_id: int):
        batch_size = int(math.ceil(float(self.size) / self.num_shards))
        return NpArrayReader(
            self._get_split(self.data, shard_id, batch_size),
            batch_size=self.batch_size,
            drop_small=self.drop_small,
        )

    def _get_split(self, data, idx, batch_size):
        if isinstance(data, OrderedDict):
            return OrderedDict(
                [(k, self._get_split(v, idx, batch_size)) for k, v in data.items()]
            )
        elif isinstance(data, np.ndarray):
            offset = idx * batch_size
            return data[offset : min(offset + batch_size, len(data))]
        else:
            raise ValueError("Got unexpected type {}".format(type(data)))

    def __iter__(self):
        return NpArrayReaderIter(self)
