#!/usr/bin/env python3

from reagent.workflow import types
from reagent.workflow.model_managers.model_manager import ModelManager

from .actor_critic import SoftActorCritic  # noqa
from .discrete import DiscreteC51DQN, DiscreteDQN, DiscreteQRDQN  # noqa


@ModelManager.fill_union()
class ModelManager__Union(types.TaggedUnion):
    pass
