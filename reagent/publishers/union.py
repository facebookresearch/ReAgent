#!/usr/bin/env python3

from reagent.workflow.types import TaggedUnion

from .file_system_publisher import FileSystemPublisher  # noqa
from .model_publisher import ModelPublisher
from .no_publishing import NoPublishing  # noqa


try:
    import fblearner.flow.projects.rl.publishing.clients  # noqa
    import fblearner.flow.projects.rl.publishing.common  # noqa
except ImportError:
    pass


@ModelPublisher.fill_union()
class ModelPublisher__Union(TaggedUnion):
    pass
