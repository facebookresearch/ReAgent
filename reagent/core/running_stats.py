#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math


class RunningStats:
    """Running statistics for elements in a stream

    Can take single values or iterables

    1. Implements Welford's algorithm for computing a running mean
    and standard deviation
    2. Min-Heap to find top-k where k < capacity (kwarg)
    Methods:
        mean    - returns the mean
        std     - returns the std
        meanfull- returns the mean and std of the mean
        topk(k) - returns the kth highest value for k < capacity
    """

    def __init__(self, lst=None, capacity: int = 1000):
        self.k = 0
        self.running_mean = 0
        self.sum_squares = 0
        self.__call__(lst)

    def update(self, x):
        if x is None:
            return
        self.k += 1
        newM = self.running_mean + (x - self.running_mean) * 1.0 / self.k
        newS = self.sum_squares + (x - self.running_mean) * (x - newM)
        self.running_mean, self.sum_squares = newM, newS

    def consume(self, lst):
        lst = iter(lst)
        for x in lst:
            self.update(x)

    def __call__(self, x):
        if hasattr(x, "__iter__"):
            self.consume(x)
        else:
            self.update(x)

    @property
    def mean(self):
        return self.running_mean

    @property
    def meanfull(self):
        return self.mean, self.std / math.sqrt(self.k)

    @property
    def std(self):
        if self.k == 1:
            return 0
        return math.sqrt(self.sum_squares / (self.k - 1))

    def __repr__(self):
        return "<Welford: {} +- {}>".format(self.mean, self.std)
