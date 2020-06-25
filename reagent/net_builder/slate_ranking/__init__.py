#!/usr/bin/env python3

from reagent.net_builder.slate_ranking_net_builder import SlateRankingNetBuilder
from reagent.workflow import types

from . import slate_ranking_transformer  # noqa


@SlateRankingNetBuilder.fill_union()
class SlateRankingNetBuilder__Union(types.TaggedUnion):
    pass
