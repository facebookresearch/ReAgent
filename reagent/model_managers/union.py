#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

""" Register all ModelManagers. Must import them before filling union. """

from typing import Optional

from reagent.core.dataclasses import dataclass
from reagent.core.tagged_union import TaggedUnion

from .actor_critic import SAC as SACType, TD3 as TD3Type
from .discrete import (
    DiscreteC51DQN as DiscreteC51DQNType,
    DiscreteCRR as DiscreteCRRType,
    DiscreteDQN as DiscreteDQNType,
    DiscreteQRDQN as DiscreteQRDQNType,
)
from .model_based import (
    CrossEntropyMethod as CrossEntropyMethodType,
    Seq2RewardModel as Seq2RewardModelType,
    WorldModel as WorldModelType,
    SyntheticReward as SyntheticRewardType,
)
from .parametric import ParametricDQN as ParametricDQNType
from .policy_gradient import PPO as PPOType, Reinforce as ReinforceType
from .ranking import SlateQ as SlateQType


@dataclass(frozen=True)
class ModelManager__Union(TaggedUnion):
    SAC: Optional[SACType] = None
    TD3: Optional[TD3Type] = None

    DiscreteC51DQN: Optional[DiscreteC51DQNType] = None
    DiscreteCRR: Optional[DiscreteCRRType] = None
    DiscreteDQN: Optional[DiscreteDQNType] = None
    DiscreteQRDQN: Optional[DiscreteQRDQNType] = None

    CrossEntropyMethod: Optional[CrossEntropyMethodType] = None
    Seq2RewardModel: Optional[Seq2RewardModelType] = None
    WorldModel: Optional[WorldModelType] = None
    SyntheticReward: Optional[SyntheticRewardType] = None

    ParametricDQN: Optional[ParametricDQNType] = None

    PPO: Optional[PPOType] = None
    Reinforce: Optional[ReinforceType] = None

    SlateQ: Optional[SlateQType] = None
