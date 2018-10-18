#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import abc
from collections import OrderedDict
from typing import Optional


class ReaderIter(object, metaclass=abc.ABCMeta):
    def __iter__(self):
        return self

    def __next__(self) -> OrderedDict:
        batch = self.read_batch()
        if batch is None:
            raise StopIteration
        return batch

    @abc.abstractmethod
    def read_batch(self) -> Optional[OrderedDict]:
        """
        Read a batch of data. The return value should be an OrderedDict.
        Returns None when there is no more data.
        """
        pass


class ReaderBase(object, metaclass=abc.ABCMeta):
    def __init__(self, batch_size=None, drop_small=True, num_shards=None):
        assert (
            batch_size is not None and batch_size > 0
        ), "batch_size should be a positive number. Got: {}".format(batch_size)
        assert num_shards is None or num_shards > 0
        self.batch_size = batch_size
        self.num_shards = num_shards
        self.drop_small = drop_small

    def get_shard(self, shard_id: int):
        """
        Returns a shard of this reader
        """
        assert self.num_shards is not None, "This reader is not shardable"
        assert (
            shard_id < self.num_shards and shard_id >= 0
        ), "Shard {} is out of range".format(shard_id)
        return self.do_get_shard(shard_id)

    def do_get_shard(self, shard_id: int):
        """
        Subclass should implement this if the reader is shardable
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self):
        """
        Returns iterator over the data. The iterator returns one batch at a time
        """
        pass
