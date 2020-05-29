#!/usr/bin/env python3

import logging
from typing import List

import reagent.optimizer.uninferrable_optimizers as cannot_be_inferred
import reagent.parameters as rlp
import torch
from reagent.core.configuration import make_config_class
from reagent.core.tagged_union import TaggedUnion
from reagent.parameters import param_hash

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
    def default(cls):
        # default factory is Adam
        return cls(Adam=classes["Adam"]())

    def make_optimizer(self, params):
        return self.value.make_optimizer(params)

    # TODO: deprecate this once we don't use OptimizerParameters anymore
    @classmethod
    def create_from_optimizer_params(cls, optimizer_params: rlp.OptimizerParameters):
        logger.warn(
            "Use registry format instead of the deprecated OptimizerParameters!"
        )
        if optimizer_params.optimizer == "ADAM":
            optimizer = classes["Adam"](
                lr=optimizer_params.learning_rate,
                weight_decay=optimizer_params.l2_decay,
            )
        elif optimizer_params.optimizer == "SGD":
            optimizer = classes["SGD"](
                lr=optimizer_params.learning_rate,
                weight_decay=optimizer_params.l2_decay,
            )
        else:
            raise NotImplementedError(f"{optimizer_params.optimizer} not supported.")
        # pyre-fixme[16]: `Optimizer__Union` has no attribute `make_union_instance`.
        return cls.make_union_instance(optimizer)
