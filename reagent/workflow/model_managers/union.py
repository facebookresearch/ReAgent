#!/usr/bin/env python3

import reagent.workflow.model_managers.discrete  # noqa
from reagent.workflow import types
from reagent.workflow.model_managers.model_manager import ModelManager


@ModelManager.fill_union()
class ModelManager__Union(types.TaggedUnion):
    pass
