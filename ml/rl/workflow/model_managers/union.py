#!/usr/bin/env python3

import ml.rl.workflow.model_managers.discrete  # noqa
from ml.rl.workflow import types
from ml.rl.workflow.model_managers.model_manager import ModelManager


@ModelManager.fill_union()
class ModelManager__Union(types.TaggedUnion):
    pass
