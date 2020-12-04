#!/usr/bin/env python3

import logging
from typing import List

import reagent.optimizer.uninferrable_schedulers as cannot_be_inferred
import torch
from reagent.core.configuration import make_config_class, param_hash
from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.tagged_union import TaggedUnion

from .scheduler import LearningRateSchedulerConfig
from .utils import is_torch_lr_scheduler


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


cannot_be_inferred_modules = [cannot_be_inferred]
if IS_FB_ENVIRONMENT:
    import reagent.optimizer.fb.uninferrable_schedulers as fb_cannot_be_inferred

    cannot_be_inferred_modules.append(fb_cannot_be_inferred)


def get_torch_lr_schedulers() -> List[str]:
    # Not type annotated and default is None (i.e unable to infer valid annotation)
    return [
        name
        for name in dir(torch.optim.lr_scheduler)
        if is_torch_lr_scheduler(getattr(torch.optim.lr_scheduler, name))
    ]


classes = {}
for name in get_torch_lr_schedulers():
    cannot_be_inferred_module = None
    for module in cannot_be_inferred_modules:
        if hasattr(module, name):
            cannot_be_inferred_module = module
            break

    if cannot_be_inferred_module is not None:
        # these were manually filled in.
        subclass = getattr(cannot_be_inferred_module, name)
    else:
        torch_lr_scheduler_class = getattr(torch.optim.lr_scheduler, name)
        subclass = type(
            name,
            # must subclass LearningRateSchedulerConfig to be added to the Registry
            (LearningRateSchedulerConfig,),
            {"__module__": __name__},
        )
        make_config_class(torch_lr_scheduler_class, blacklist=["optimizer"])(subclass)

    subclass.__hash__ = param_hash
    classes[name] = subclass


@LearningRateSchedulerConfig.fill_union()
class LearningRateScheduler__Union(TaggedUnion):
    pass
