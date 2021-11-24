#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional

from reagent.core.registry_meta import wrap_oss_with_dataclass
from reagent.core.tagged_union import TaggedUnion

from .slate_reward_gru import SlateRewardGRU as SlateRewardGRUType
from .slate_reward_transformer import (
    SlateRewardTransformer as SlateRewardTransformerType,
)


@wrap_oss_with_dataclass
class SlateRewardNetBuilder__Union(TaggedUnion):
    SlateRewardGRU: Optional[SlateRewardGRUType] = None
    SlateRewardTransformer: Optional[SlateRewardTransformerType] = None
