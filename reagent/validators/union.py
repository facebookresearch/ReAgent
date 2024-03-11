#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# pyre-unsafe

from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.tagged_union import TaggedUnion

from .model_validator import ModelValidator
from .no_validation import NoValidation  # noqa


if IS_FB_ENVIRONMENT:
    # pyre-fixme[21]: Could not find module
    #  `fblearner.flow.projects.rl.validation.clients`.
    import fblearner.flow.projects.rl.validation.clients  # noqa

    # pyre-fixme[21]: Could not find module
    #  `fblearner.flow.projects.rl.validation.common`.
    import fblearner.flow.projects.rl.validation.common  # noqa


@ModelValidator.fill_union()
class ModelValidator__Union(TaggedUnion):
    pass
