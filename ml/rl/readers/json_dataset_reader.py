#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import bz2
import gzip
import math
import os
from collections import OrderedDict
from typing import Optional

import pandas as pd
from ml.rl.readers.base import ReaderBase, ReaderIter


class JSONDatasetReaderIter(ReaderIter):
    def __init__(self, reader):
        self.reader = reader
        self.batch_read = 0
        self.total_batches = (
            reader.len // reader.batch_size
            if reader.drop_small
            else int(math.ceil(reader.len / reader.batch_size))
        )

    def read_batch(self) -> Optional[OrderedDict]:
        if self.batch_read == self.total_batches:
            return None
        ret = OrderedDict(self.reader.read_batch())
        self.batch_read += 1
        return ret


class JSONDatasetReader(ReaderBase):
    """Create the reader for a JSON training dataset."""

    def __init__(self, path, batch_size=None, converter=None):
        super().__init__(batch_size=batch_size)
        self.path = os.path.expanduser(path)
        self.file_type = path.split(".")[-1]
        self.len = self.line_count()
        self.reset_iterator()

    def reset_iterator(self):
        self.data_iterator = pd.read_json(
            self.path, lines=True, chunksize=self.batch_size
        )

    def read_batch(self):
        assert (
            self.batch_size is not None
        ), "Batch size must be provided to read data in batches."

        try:
            x = next(self.data_iterator)
        except StopIteration:
            # No more data to read
            return None
        return x.to_dict(orient="list")

    def read_all(self):
        return pd.read_json(self.path, lines=True).to_dict(orient="list")

    def __len__(self):
        return self.len

    def line_count(self):
        lines = 0
        if self.file_type == "gz":
            with gzip.open(self.path) as f:
                for _ in f:
                    lines += 1
        elif self.file_type == "bz2":
            with bz2.open(self.path) as f:
                for _ in f:
                    lines += 1
        else:
            with open(self.path) as f:
                for _ in f:
                    lines += 1
        return lines

    def __iter__(self):
        return JSONDatasetReaderIter(self)
