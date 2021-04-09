#!/usr/bin/env python3

from reagent.core.tagged_union import TaggedUnion
from reagent.net_builder.slate_ranking_net_builder import SlateRankingNetBuilder

from . import slate_ranking_scorer  # noqa
from . import slate_ranking_transformer  # noqa


@SlateRankingNetBuilder.fill_union()
class SlateRankingNetBuilder__Union(TaggedUnion):
    pass
