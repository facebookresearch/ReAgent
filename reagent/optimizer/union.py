#!/usr/bin/env python3

import logging
from typing import List

import reagent.optimizer.uninferrable_optimizers as cannot_be_inferred
import torch
from reagent.core.configuration import make_config_class, param_hash
from reagent.core.tagged_union import TaggedUnion

from .optimizer import OptimizerConfig
from .utils import is_torch_optimizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_torch_optimizers() -> List[str]:
    return [
        name
        for name in dir(torch.optim)
        if is_torch_optimizer(getattr(torch.optim, name))
    ]


classes = {}
for name in get_torch_optimizers():
    if hasattr(cannot_be_inferred, name):
        # these were manually filled in.
        subclass = getattr(cannot_be_inferred, name)
    else:
        # this points to the optimizer class in torch.optim (e.g. Adam)
        torch_optimizer_class = getattr(torch.optim, name)

        # dynamically create wrapper class, which has the same name as torch_class
        subclass = type(
            name,
            # must subclass Optimizer to be added to the Registry
            (OptimizerConfig,),
            {},
        )
        # fill in optimizer parameters (except params)
        make_config_class(torch_optimizer_class, blacklist=["params"])(subclass)

    subclass.__hash__ = param_hash
    classes[name] = subclass


@OptimizerConfig.fill_union()
class Optimizer__Union(TaggedUnion):
    @classmethod
    def default(cls, **kwargs):
        """ Return default factory for Optimizer (defaulting to Adam). """
        return (
            cls(Adam=classes["Adam"]())
            if kwargs == {}
            else cls(Adam=classes["Adam"](**kwargs))
        )

    def make_optimizer(self, params):
        return self.value.make_optimizer(params)
