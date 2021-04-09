#!/usr/bin/env python3

from typing import Optional

from reagent.core.registry_meta import wrap_oss_with_dataclass
from reagent.core.tagged_union import TaggedUnion

from .slate_ranking_transformer import (
    SlateRankingTransformer as SlateRankingTransformerType,
)


@wrap_oss_with_dataclass
class SlateRankingNetBuilder__Union(TaggedUnion):
    SlateRankingTransformer: Optional[SlateRankingTransformerType] = None
