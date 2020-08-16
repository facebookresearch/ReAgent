#!/usr/bin/env python3

from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.tagged_union import TaggedUnion

from .model_validator import ModelValidator
from .no_validation import NoValidation  # noqa


if IS_FB_ENVIRONMENT:
    import fblearner.flow.projects.rl.validation.clients  # noqa


@ModelValidator.fill_union()
class ModelValidator__Union(TaggedUnion):
    pass
