#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import inspect

import torch


def is_strict_subclass(a, b):
    if not inspect.isclass(a) or not inspect.isclass(b):
        return False
    return issubclass(a, b) and a != b


def is_torch_optimizer(cls):
    return is_strict_subclass(cls, torch.optim.Optimizer)


def is_torch_lr_scheduler(cls):
    return is_strict_subclass(cls, torch.optim.lr_scheduler._LRScheduler)
