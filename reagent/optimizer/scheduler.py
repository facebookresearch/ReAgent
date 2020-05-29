#!/usr/bin/env python3

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
        assert is_torch_lr_scheduler(
            torch_lr_scheduler_class
        ), f"{torch_lr_scheduler_class} is not a scheduler."
        return torch_lr_scheduler_class(optimizer=optimizer, **vars(self))
