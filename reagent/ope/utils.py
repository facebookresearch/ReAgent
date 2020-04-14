#!/usr/bin/env python3

from collections import OrderedDict
from typing import Sequence, Union

import numpy as np
import torch


def convert_to_one_hots(a, num_classes: int, dtype=torch.int, device=None):
    """
    Convert class index array (num_sample,) to an one hots array
    (num_sample, num_classes)

    Args:
        a: index array
        num_classes: number of classes
        dtype: data type

    Returns:
        one hots array in shape of (a.shape[0], num_classes)
    """
    one_hots = torch.zeros((len(a), num_classes), dtype=dtype, device=device)
    one_hots[torch.arange(one_hots.shape[0]), a] = 1
    return one_hots


class LRUCache(OrderedDict):
    def __init__(self, maxsize=2 ** 10, *args, **kwds):
        self.maxsize = maxsize
        super().__init__(*args, **kwds)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            del self[next(iter(self))]


class RunningAverage:
    def __init__(self):
        self._average = 0.0
        self._count = 0

    def add(self, value) -> "RunningAverage":
        self._count += 1
        self._average = self._average + (float(value) - self._average) / self._count
        return self

    @property
    def average(self):
        return self._average

    @property
    def count(self):
        return self._count

    @property
    def total(self):
        return self._average * self._count


class Clamper:
    def __init__(self, min: float = None, max: float = None):
        self._min = min if min is not None else float("-inf")
        self._max = max if max is not None else float("inf")
        if self._min >= self._max:
            raise ValueError(f"min[{min}] greater than max[{max}]")

    def __call__(
        self, v: Union[float, Sequence[float], torch.Tensor, np.ndarray]
    ) -> Union[float, Sequence[float], torch.Tensor, np.ndarray]:
        if isinstance(v, torch.Tensor):
            return v.clamp(self._min, self._max)
        elif isinstance(v, np.ndarray):
            return v.clip(self._min, self._max)
        elif isinstance(v, Sequence):
            return [max(self._min, min(self._max, float(i))) for i in v]
        else:
            return max(self._min, min(self._max, float(v)))
