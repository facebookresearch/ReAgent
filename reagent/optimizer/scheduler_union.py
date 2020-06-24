#!/usr/bin/env python3

import logging
from typing import List

import reagent.optimizer.uninferrable_schedulers as cannot_be_inferred
import torch
from reagent.core.configuration import make_config_class, param_hash
from reagent.core.tagged_union import TaggedUnion

from .scheduler import LearningRateSchedulerConfig
from .utils import is_torch_lr_scheduler


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_torch_lr_schedulers() -> List[str]:
    # Not type annotated and default is None (i.e unable to infer valid annotation)
    return [
        name
        for name in dir(torch.optim.lr_scheduler)
        if is_torch_lr_scheduler(getattr(torch.optim.lr_scheduler, name))
    ]


classes = {}
for name in get_torch_lr_schedulers():
    if hasattr(cannot_be_inferred, name):
        # these were manually filled in.
        subclass = getattr(cannot_be_inferred, name)
    else:
        torch_lr_scheduler_class = getattr(torch.optim.lr_scheduler, name)
        subclass = type(
            name,
            # must subclass Optimizer to be added to the Registry
            (LearningRateSchedulerConfig,),
            {"__module__": __name__},
        )
        make_config_class(torch_lr_scheduler_class, blacklist=["optimizer"])(subclass)

    subclass.__hash__ = param_hash
    classes[name] = subclass


@LearningRateSchedulerConfig.fill_union()
class LearningRateScheduler__Union(TaggedUnion):
    def make_from_optimizer(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return self.value.make_from_optimizer(optimizer)
