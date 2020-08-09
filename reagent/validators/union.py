#!/usr/bin/env python3

from reagent.workflow.types import TaggedUnion

from .model_validator import ModelValidator
from .no_validation import NoValidation  # noqa


try:
    import fblearner.flow.projects.rl.validation.clients  # noqa
except ImportError:
    pass


@ModelValidator.fill_union()
class ModelValidator__Union(TaggedUnion):
    pass
