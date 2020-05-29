#!/usr/bin/env python3

"""
For each Torch optimizer, we create a wrapper pydantic dataclass around it.
We also add this class to our Optimizer registry.

Usage:

Whenever you want to use this Optimizer__Union, specify it as the type.
E.g.
class Parameters:
    rl: RLParameters = field(default_factory=RLParameters)
    minibatch_size: int = 64
    optimizer: Optimizer__Union = field(default_factory=Optimizer__Union.default)

To instantiate it, specify desired optimzier in YAML file.
E.g.
rl:
  ...
minibatch: 64
optimizer:
  Adam:
    lr: 0.001
    eps: 1e-08
    lr_schedulers:
        - OneCycleLR:
            ...

Since we don't know which network parameters we want to optimize,
Optimizer__Union will be a factory for the optimizer it contains.

Following the above example, we create an optimizer as follows:

class Trainer:
    def __init__(self, network, params):
        self.optimizer = params.optimizer.make_optimizer(network.parameters())

    def train(self, data):
        ...
        loss.backward()
        # steps both optimizer and chained lr_schedulers
        self.optimizer.step()
"""
import inspect
from typing import List

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.registry_meta import RegistryMeta

from .scheduler_union import LearningRateScheduler__Union
from .utils import is_torch_optimizer


@dataclass(frozen=True)
class Optimizer:
    # This is the wrapper for optimizer + scheduler
    optimizer: torch.optim.Optimizer
    lr_schedulers: List[torch.optim.lr_scheduler._LRScheduler]

    def step(self):
        self.optimizer.step()
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

    def __getattr__(self, attr):
        return getattr(self.optimizer, attr)


@dataclass(frozen=True)
class OptimizerConfig(metaclass=RegistryMeta):
    # optional config if you want to use (potentially chained) lr scheduler
    lr_schedulers: List[LearningRateScheduler__Union] = field(default_factory=list)

    def make_optimizer(self, params) -> Optimizer:
        # Assuming the classname is the same as the torch class name
        torch_optimizer_class = getattr(torch.optim, type(self).__name__)
        assert is_torch_optimizer(
            torch_optimizer_class
        ), f"{torch_optimizer_class} is not an optimizer."
        filtered_args = {
            k: getattr(self, k)
            for k in inspect.signature(torch_optimizer_class).parameters
            if k != "params"
        }
        optimizer = torch_optimizer_class(params=params, **filtered_args)
        lr_schedulers = [
            lr_scheduler.make_from_optimizer(optimizer)
            for lr_scheduler in self.lr_schedulers
        ]
        return Optimizer(optimizer=optimizer, lr_schedulers=lr_schedulers)
