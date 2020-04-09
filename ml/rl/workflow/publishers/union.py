#!/usr/bin/env python3

from ml.rl.workflow import types
from ml.rl.workflow.publishers.no_publishing import NoPublishing  # noqa

from .model_publisher import ModelPublisher


@ModelPublisher.fill_union()
class ModelPublisher__Union(types.TaggedUnion):
    pass
