#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

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
        self.optimizer = params.optimizer.make_optimizer_scheduler(network.parameters())["optimizer"]

    def train(self, data):
        ...
        loss.backward()
        # steps both optimizer and chained lr_schedulers
        self.optimizer.step()
"""
import inspect
from typing import Dict, List, Union

import torch
from reagent.core.dataclasses import dataclass, field
from reagent.core.registry_meta import RegistryMeta

from .scheduler import LearningRateSchedulerConfig
from .utils import is_torch_optimizer


@dataclass(frozen=True)
class OptimizerConfig(metaclass=RegistryMeta):
    # optional config if you want to use (potentially chained) lr scheduler
    lr_schedulers: List[LearningRateSchedulerConfig] = field(default_factory=list)

    def make_optimizer_scheduler(
        self, params
    ) -> Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]]:
        assert (
            len(self.lr_schedulers) <= 1
        ), "Multiple schedulers for one optimizer is no longer supported"
        # Assuming the classname is the same as the torch class name
        torch_optimizer_class = getattr(torch.optim, type(self).__name__)
        assert is_torch_optimizer(
            torch_optimizer_class
        ), f"{torch_optimizer_class} is not an optimizer."
        filtered_args = {
            k: getattr(self, k)
            for k in inspect.signature(torch_optimizer_class).parameters
            if k != "params" and hasattr(self, k)
        }
        optimizer = torch_optimizer_class(params=params, **filtered_args)
        if len(self.lr_schedulers) == 0:
            return {"optimizer": optimizer}
        else:
            lr_scheduler = self.lr_schedulers[0].make_from_optimizer(optimizer)
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
