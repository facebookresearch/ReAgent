#!/usr/bin/env python3

from reagent.workflow import types

from .model_validator import ModelValidator
from .no_validation import NoValidation  # noqa


@ModelValidator.fill_union()
class ModelValidator__Union(types.TaggedUnion):
    pass
