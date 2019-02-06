#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import bz2
import gzip
import json
import os
from typing import Dict

import pandas as pd
from ml.rl.preprocessing import normalization
from ml.rl.thrift.core.ttypes import NormalizationParameters


class JSONDataset:
    """Create the reader for a JSON training dataset."""

    def __init__(self, path, batch_size=None, converter=None):
        self.path = os.path.expanduser(path)
        self.file_type = path.split(".")[-1]
        self.batch_size = batch_size
        self.len = self.line_count()
        self.reset_iterator()

    def reset_iterator(self):
        self.data_iterator = pd.read_json(
            self.path, lines=True, chunksize=self.batch_size
        )

    def read_batch(self, astype="dict"):
        assert (
            self.batch_size is not None
        ), "Batch size must be provided to read data in batches."

        try:
            x = next(self.data_iterator)
        except StopIteration:
            # No more data to read
            return None
        if astype == "dict":
            return x.to_dict(orient="list")
        return x

    def read_all(self):
        return pd.read_json(self.path, lines=True)

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


def read_norm_file(path) -> Dict[int, NormalizationParameters]:
    path = os.path.expanduser(path)
    if path.split(".")[-1] == "gz":
        with gzip.open(path) as f:
            norm_json = json.load(f)
    else:
        with open(path) as f:
            norm_json = json.load(f)
    return normalization.deserialize(norm_json)
