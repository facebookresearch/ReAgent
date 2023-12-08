#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.tagged_union import TaggedUnion

from .file_system_publisher import FileSystemPublisher  # noqa
from .model_publisher import ModelPublisher
from .no_publishing import NoPublishing  # noqa


if IS_FB_ENVIRONMENT:
    # pyre-fixme[21]: Could not find module
    #  `fblearner.flow.projects.rl.publishing.clients`.
    import fblearner.flow.projects.rl.publishing.clients  # noqa

    # pyre-fixme[21]: Could not find module
    #  `fblearner.flow.projects.rl.publishing.common`.
    import fblearner.flow.projects.rl.publishing.common  # noqa


@ModelPublisher.fill_union()
class ModelPublisher__Union(TaggedUnion):
    pass
