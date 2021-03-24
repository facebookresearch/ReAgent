#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

""" Register all ModelManagers. Must import them before filling union. """

from reagent.core.tagged_union import TaggedUnion
from reagent.model_managers.model_manager import ModelManager

from .actor_critic import *  # noqa
from .discrete import *  # noqa
from .model_based import *  # noqa
from .parametric import *  # noqa
from .policy_gradient import *  # noqa
from .ranking import *  # noqa


@ModelManager.fill_union()
class ModelManager__Union(TaggedUnion):
    pass
