#!/usr/bin/env python3

try:
    from fblearner.flow.api import types
except ImportError:
    from ml.rl.polyfill import types  # type: ignore

from . import categorical_dqn  # noqa
from . import continuous_actor  # noqa
from . import discrete_dqn  # noqa
from . import parametric_dqn  # noqa
from . import quantile_dqn  # noqa
from . import value  # noqa
from .categorical_dqn_net_builder import CategoricalDQNNetBuilder
from .continuous_actor_net_builder import ContinuousActorNetBuilder
from .discrete_dqn_net_builder import DiscreteDQNNetBuilder
from .parametric_dqn_net_builder import ParametricDQNNetBuilder
from .quantile_dqn_net_builder import QRDQNNetBuilder
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


class CategoricalDQNNetBuilderChooser(types.TaggedUnion):
    pass


CategoricalDQNNetBuilderChooser.__annotations__ = {
    name: t.config_type() for name, t in CategoricalDQNNetBuilder.REGISTRY.items()
}


class QRDQNNetBuilderChooser(types.TaggedUnion):
    pass


QRDQNNetBuilderChooser.__annotations__ = {
    name: t.config_type() for name, t in QRDQNNetBuilder.REGISTRY.items()
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
