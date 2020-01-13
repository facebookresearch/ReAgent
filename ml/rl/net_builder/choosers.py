#!/usr/bin/env python3

try:
    from fblearner.flow.api import types
except ImportError:
    from ml.rl.polyfill import types  # type: ignore

from . import continuous_actor  # noqa
from . import discrete_dqn  # noqa
from . import parametric_dqn  # noqa
from . import value  # noqa
from .continuous_actor_net_builder import ContinuousActorNetBuilder
from .discrete_dqn_net_builder import DiscreteDQNNetBuilder
from .parametric_dqn_net_builder import ParametricDQNNetBuilder
from .value_net_builder import ValueNetBuilder


class ContinuousActorNetBuilderChooser(types.TaggedUnion):
    pass


ContinuousActorNetBuilderChooser.__annotations__ = {
    name: t.config_type() for name, t in ContinuousActorNetBuilder.REGISTRY.items()
}


class DiscreteDQNNetBuilderChooser(types.TaggedUnion):
    pass


DiscreteDQNNetBuilderChooser.__annotations__ = {
    name: t.config_type() for name, t in DiscreteDQNNetBuilder.REGISTRY.items()
}


class ParametricDQNNetBuilderChooser(types.TaggedUnion):
    pass


ParametricDQNNetBuilderChooser.__annotations__ = {
    name: t.config_type() for name, t in ParametricDQNNetBuilder.REGISTRY.items()
}


class ValueNetBuilderChooser(types.TaggedUnion):
    pass


ValueNetBuilderChooser.__annotations__ = {
    name: t.config_type() for name, t in ValueNetBuilder.REGISTRY.items()
}
