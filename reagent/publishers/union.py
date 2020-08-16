#!/usr/bin/env python3

from reagent.core.fb_checker import IS_FB_ENVIRONMENT
from reagent.core.tagged_union import TaggedUnion

from .file_system_publisher import FileSystemPublisher  # noqa
from .model_publisher import ModelPublisher
from .no_publishing import NoPublishing  # noqa


if IS_FB_ENVIRONMENT:
    import fblearner.flow.projects.rl.publishing.clients  # noqa
    import fblearner.flow.projects.rl.publishing.common  # noqa


@ModelPublisher.fill_union()
class ModelPublisher__Union(TaggedUnion):
    pass
