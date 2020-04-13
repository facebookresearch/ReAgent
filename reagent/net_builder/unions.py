#!/usr/bin/env python3

from reagent.workflow import types

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


@ContinuousActorNetBuilder.fill_union()
class ContinuousActorNetBuilder__Union(types.TaggedUnion):
    pass


@DiscreteDQNNetBuilder.fill_union()
class DiscreteDQNNetBuilder__Union(types.TaggedUnion):
    pass


@CategoricalDQNNetBuilder.fill_union()
class CategoricalDQNNetBuilder__Union(types.TaggedUnion):
    pass


@QRDQNNetBuilder.fill_union()
class QRDQNNetBuilder__Union(types.TaggedUnion):
    pass


@ParametricDQNNetBuilder.fill_union()
class ParametricDQNNetBuilder__Union(types.TaggedUnion):
    pass


@ValueNetBuilder.fill_union()
class ValueNetBuilder__Union(types.TaggedUnion):
    pass
