#!/usr/bin/env python3

from reagent.net_builder.slate_reward_net_builder import SlateRewardNetBuilder
from reagent.workflow import types

from . import slate_reward_gru  # noqa
from . import slate_reward_transformer  # noqa


@SlateRewardNetBuilder.fill_union()
class SlateRewardNetBuilder__Union(types.TaggedUnion):
    pass
