#!/usr/bin/env python3

from reagent.workflow import types

from .file_system_publisher import FileSystemPublisher  # noqa
from .model_publisher import ModelPublisher
from .no_publishing import NoPublishing  # noqa


@ModelPublisher.fill_union()
class ModelPublisher__Union(types.TaggedUnion):
    pass
