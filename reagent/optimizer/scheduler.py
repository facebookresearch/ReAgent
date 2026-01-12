#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

import inspect
from typing import Any, Dict

import torch
from reagent.core.dataclasses import dataclass
from reagent.core.registry_meta import RegistryMeta

from .utils import is_torch_lr_scheduler


@dataclass(frozen=True)
class LearningRateSchedulerConfig(metaclass=RegistryMeta):
    def make_from_optimizer(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        torch_lr_scheduler_class = getattr(
            torch.optim.lr_scheduler, type(self).__name__
        )
        assert is_torch_lr_scheduler(torch_lr_scheduler_class), (
            f"{torch_lr_scheduler_class} is not a scheduler."
        )

        filtered_args = {
            k: getattr(self, k)
            for k in inspect.signature(torch_lr_scheduler_class).parameters
            if k != "optimizer"
        }

        self.decode_lambdas(filtered_args)

        return torch_lr_scheduler_class(optimizer=optimizer, **filtered_args)

    def decode_lambdas(self, args: Dict[str, Any]) -> None:
        pass
